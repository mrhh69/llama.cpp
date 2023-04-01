
#ifndef SERVER_H
#define SERVER_H

// size of buffer for prompt input from socket
#define BUF_SIZE 2048
// select() wait times
#define IDLE_WAIT_S 0
#define IDLE_WAIT_US 100000
// port/queue_size for listen() call
#define PORT 3002
#define QUEUE_SIZE 10


#ifdef __cplusplus
extern "C" {
#endif
extern void new_token_str(const char *);
extern void server_loop(void);
extern void process(char *);

#ifdef __cplusplus
}
#endif

#endif
