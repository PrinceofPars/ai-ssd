#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <signal.h>
#include <immintrin.h>

#define MAGIC 0x4E564D45U /* 'NVME' in ASCII big-endian */
#define OP_WRITE 1
#define OP_READ 2
#define OP_PING 3
#define OP_FLUSH 4
#define OP_SHUTDOWN 5
#define OP_BATCH_READ 6
#define OP_COMPUTE_TOPK 7

#define BUFFER_SIZE (1024 * 1024) /* 1 MB static buffer */

#pragma pack(push, 1)
struct req_header {
    uint32_t magic;
    uint8_t op;
    uint8_t flags;
    uint16_t reserved;
    uint64_t offset;
    uint32_t length;
};

struct resp_header {
    uint32_t magic;
    uint8_t status;
    uint8_t op;
    uint16_t reserved;
    uint64_t offset;
    uint32_t length;
};

struct batch_read_item {
    uint64_t offset;
    uint32_t length;
    uint32_t block_id;
};

struct topk_req_header {
    uint32_t num_candidates;
    uint32_t top_k;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float scale;
};

struct topk_cand_item {
    uint64_t offset;
    uint32_t length;
    uint32_t block_id;
    uint32_t actual_tokens;
};

struct topk_resp_item {
    uint32_t block_id;
    float score;
    uint32_t actual_tokens;
};
#pragma pack(pop)

static inline float dot_product_64_avx2(const float* a, const float* b) {
    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc0);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static inline float compute_block_score_gqa(
    const float* query,
    const float* k_block,
    int actual_tokens,
    int q_heads,
    int kv_heads,
    int head_dim,
    float scale
) {
    float max_score = -1e30f;
    int gqa_ratio = (kv_heads > 0) ? (q_heads / kv_heads) : 1;
    int kv_stride = kv_heads * head_dim;

    for (int t = 0; t < actual_tokens; t++) {
        const float* k_token = k_block + (t * kv_stride);
        for (int qh = 0; qh < q_heads; qh++) {
            int kh = qh / gqa_ratio;
            const float* q_vec = query + (qh * head_dim);
            const float* k_vec = k_token + (kh * head_dim);
            float dot;
            if (head_dim == 64) {
                dot = dot_product_64_avx2(q_vec, k_vec);
            } else {
                dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
            }
            float scaled_dot = dot * scale;
            if (scaled_dot > max_score) {
                max_score = scaled_dot;
            }
        }
    }
    return max_score;
}

static int read_all(int fd, void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, (char *)buf + total, len - total);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            return -1;
        }
        total += (size_t)n;
    }
    return 0;
}

static int write_all(int fd, const void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = write(fd, (const char *)buf + total, len - total);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            return -1;
        }
        total += (size_t)n;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *dev_path = (argc > 1) ? argv[1] : "/dev/nvme0n1";
    int port = (argc > 2) ? atoi(argv[2]) : 9999;

    signal(SIGPIPE, SIG_IGN);

    int dev_fd = open(dev_path, O_RDWR);
    if (dev_fd < 0) {
        fprintf(stderr, "[NVME_DAEMON] Failed to open %s: %s\n", dev_path, strerror(errno));
        return 1;
    }

    int s_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (s_fd < 0) {
        fprintf(stderr, "[NVME_DAEMON] Failed to create socket: %s\n", strerror(errno));
        close(dev_fd);
        return 1;
    }

    int opt = 1;
    setsockopt(s_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(s_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[NVME_DAEMON] Failed to bind to port %d: %s\n", port, strerror(errno));
        close(s_fd);
        close(dev_fd);
        return 1;
    }

    if (listen(s_fd, 8) < 0) {
        fprintf(stderr, "[NVME_DAEMON] Failed to listen: %s\n", strerror(errno));
        close(s_fd);
        close(dev_fd);
        return 1;
    }

    fprintf(stdout, "[NVME_DAEMON] Listening on port %d, device: %s\n", port, dev_path);
    fflush(stdout);

    static char io_buf[BUFFER_SIZE];
    int running = 1;

    while (running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int c_fd = accept(s_fd, (struct sockaddr *)&client_addr, &client_len);
        if (c_fd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        int nodelay = 1;
        setsockopt(c_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        while (1) {
            struct req_header req;
            if (read_all(c_fd, &req, sizeof(req)) < 0) {
                break; /* Client disconnected */
            }

            if (req.magic != MAGIC) {
                fprintf(stderr, "[NVME_DAEMON] Invalid magic: 0x%08X\n", req.magic);
                break;
            }

            struct resp_header resp;
            memset(&resp, 0, sizeof(resp));
            resp.magic = MAGIC;
            resp.op = req.op;
            resp.offset = req.offset;

            if (req.op == OP_PING) {
                resp.status = 0;
                resp.length = 0;
                write_all(c_fd, &resp, sizeof(resp));
            } else if (req.op == OP_WRITE) {
                if (req.length > BUFFER_SIZE) {
                    resp.status = 1;
                    resp.length = 0;
                    write_all(c_fd, &resp, sizeof(resp));
                    continue;
                }
                if (read_all(c_fd, io_buf, req.length) < 0) {
                    break;
                }
                ssize_t pw = pwrite(dev_fd, io_buf, req.length, (off_t)req.offset);
                if (pw < 0 || (size_t)pw != req.length) {
                    resp.status = 1;
                    resp.length = (pw > 0) ? (uint32_t)pw : 0;
                } else {
                    resp.status = 0;
                    resp.length = (uint32_t)pw;
                }
                write_all(c_fd, &resp, sizeof(resp));
            } else if (req.op == OP_READ) {
                if (req.length > BUFFER_SIZE) {
                    resp.status = 1;
                    resp.length = 0;
                    write_all(c_fd, &resp, sizeof(resp));
                    continue;
                }
                ssize_t pr = pread(dev_fd, io_buf, req.length, (off_t)req.offset);
                if (pr < 0) {
                    resp.status = 1;
                    resp.length = 0;
                    write_all(c_fd, &resp, sizeof(resp));
                } else {
                    resp.status = 0;
                    resp.length = (uint32_t)pr;
                    write_all(c_fd, &resp, sizeof(resp));
                    if (pr > 0) {
                        write_all(c_fd, io_buf, (size_t)pr);
                    }
                }
            } else if (req.op == OP_BATCH_READ) {
                /* req.length contains number of items */
                uint32_t num_items = req.length;
                struct batch_read_item *items = (struct batch_read_item *)malloc(num_items * sizeof(struct batch_read_item));
                if (!items || read_all(c_fd, items, num_items * sizeof(struct batch_read_item)) < 0) {
                    free(items);
                    break;
                }
                resp.status = 0;
                resp.length = num_items;
                write_all(c_fd, &resp, sizeof(resp));

                for (uint32_t i = 0; i < num_items; i++) {
                    uint32_t item_len = items[i].length;
                    uint32_t bid = items[i].block_id;
                    ssize_t pr = pread(dev_fd, io_buf, item_len, (off_t)items[i].offset);
                    uint8_t item_status = (pr == (ssize_t)item_len) ? 0 : 1;
                    write_all(c_fd, &item_status, 1);
                    write_all(c_fd, &bid, 4);
                    write_all(c_fd, &item_len, 4);
                    if (item_status == 0) {
                        write_all(c_fd, io_buf, item_len);
                    }
                }
                free(items);
            } else if (req.op == OP_COMPUTE_TOPK) {
                /* In-storage Computational Top-K filtering */
                struct topk_req_header theader;
                if (read_all(c_fd, &theader, sizeof(theader)) < 0) {
                    break;
                }
                uint32_t num_cands = theader.num_candidates;
                uint32_t top_k = theader.top_k;
                uint32_t q_elements = theader.q_heads * theader.head_dim;
                float *query_buf = (float *)malloc(q_elements * sizeof(float));
                struct topk_cand_item *cands = (struct topk_cand_item *)malloc(num_cands * sizeof(struct topk_cand_item));

                if (!query_buf || !cands ||
                    read_all(c_fd, query_buf, q_elements * sizeof(float)) < 0 ||
                    read_all(c_fd, cands, num_cands * sizeof(struct topk_cand_item)) < 0) {
                    free(query_buf);
                    free(cands);
                    break;
                }

                uint32_t effective_k = (top_k < num_cands) ? top_k : num_cands;
                struct topk_resp_item *top_items = (struct topk_resp_item *)calloc(effective_k, sizeof(struct topk_resp_item));
                for (uint32_t k = 0; k < effective_k; k++) {
                    top_items[k].score = -1e30f;
                    top_items[k].block_id = 0;
                    top_items[k].actual_tokens = 0;
                }

                /* In-storage execution: read candidate K pages directly from NVMe and compute dot-product scores */
                for (uint32_t i = 0; i < num_cands; i++) {
                    uint32_t item_len = cands[i].length;
                    if (item_len > BUFFER_SIZE) item_len = BUFFER_SIZE;
                    ssize_t pr = pread(dev_fd, io_buf, item_len, (off_t)cands[i].offset);
                    if (pr != (ssize_t)item_len) continue;

                    float score = compute_block_score_gqa(
                        query_buf, (const float *)io_buf, (int)cands[i].actual_tokens,
                        (int)theader.q_heads, (int)theader.kv_heads, (int)theader.head_dim, theader.scale
                    );

                    if (effective_k > 0 && score > top_items[effective_k - 1].score) {
                        int insert_pos = (int)effective_k - 1;
                        while (insert_pos > 0 && score > top_items[insert_pos - 1].score) {
                            insert_pos--;
                        }
                        for (int j = (int)effective_k - 1; j > insert_pos; j--) {
                            top_items[j] = top_items[j - 1];
                        }
                        top_items[insert_pos].block_id = cands[i].block_id;
                        top_items[insert_pos].score = score;
                        top_items[insert_pos].actual_tokens = cands[i].actual_tokens;
                    }
                }

                resp.status = 0;
                resp.length = effective_k;
                write_all(c_fd, &resp, sizeof(resp));
                if (effective_k > 0) {
                    write_all(c_fd, top_items, effective_k * sizeof(struct topk_resp_item));
                }

                free(query_buf);
                free(cands);
                free(top_items);
            } else if (req.op == OP_FLUSH) {
                fdatasync(dev_fd);
                resp.status = 0;
                resp.length = 0;
                write_all(c_fd, &resp, sizeof(resp));
            } else if (req.op == OP_SHUTDOWN) {
                resp.status = 0;
                resp.length = 0;
                write_all(c_fd, &resp, sizeof(resp));
                running = 0;
                close(c_fd);
                break;
            }
        }
        close(c_fd);
    }

    close(s_fd);
    close(dev_fd);
    fprintf(stdout, "[NVME_DAEMON] Daemon terminated cleanly.\n");
    return 0;
}
