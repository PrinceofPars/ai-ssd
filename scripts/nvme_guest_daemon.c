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

#define MAGIC 0x4E564D45U /* 'NVME' in ASCII big-endian */
#define OP_WRITE 1
#define OP_READ 2
#define OP_PING 3
#define OP_FLUSH 4
#define OP_SHUTDOWN 5
#define OP_BATCH_READ 6

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
#pragma pack(pop)

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
