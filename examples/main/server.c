#include "server.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <errno.h>

#include <sys/select.h>
#include <sys/socket.h>
#include <arpa/inet.h>



int cur_sock; // only used on handler

int cur_pid;



void new_token_str(const char *s) {
	//printf("%s", s);
	//fflush(stdout);
	write(cur_sock, s, strlen(s));
}


void handler_cleanup(int sig) {
	printf("sig: %i\n", sig);
	close(cur_sock);
	exit(0);
}


static void accepthandler(int sock) {
	cur_sock = sock;

	char buf[BUF_SIZE];
	int i = 0;
	do {
		read(sock, &buf[i], 1);
	} while (buf[i++] && i < BUF_SIZE - 1);
	buf[i] = 0;

	printf("prompt: '%s'\n", buf);

	process(&buf[0]);


	close(cur_sock);
}

int sock; // used on server

void cleanup(int s) {
	printf("sig: %i\n", s);
	if (cur_pid) kill(cur_pid, SIGINT);
	close(sock);
	exit(0);
}

// returns 0 if a connection is ready
static int idle() {
	struct timeval tv = {IDLE_WAIT_S, IDLE_WAIT_US};
	fd_set set;
	FD_ZERO(&set);
	FD_SET(sock, &set);

	if (select(FD_SETSIZE, &set, NULL, NULL, &tv) == -1) {printf("select\n"); exit(1);}
	if (cur_pid) {
		// see if child process exited yet
		int stat;
		switch (waitpid(cur_pid, &stat, WNOHANG)) {
			case -1: printf("waitpid: %i\n", errno); exit(1);
			case 0: break;// no children exited yet
			default:
				// child exited
				printf("child %i exited (%i)\n", cur_pid, WEXITSTATUS(stat));
				cur_pid = 0;
		}
	}
	if (FD_ISSET(sock, &set)) return 0;
	return 1;
}


void server_loop(void) {
	struct sockaddr_in addr;

	if ((sock = socket(PF_INET, SOCK_STREAM, 0)) == -1) {printf("socket\n"); exit(1);}

	addr.sin_family = AF_INET;
  addr.sin_port = htons(PORT);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

	if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {printf("bind: %i\n", errno); exit(1);}

	if (listen(sock, QUEUE_SIZE) == -1) {printf("listen\n"); exit(1);}

	signal(SIGINT, &cleanup);

	for (;;) {
		int acc;
		char c;
		printf("\nServer listening for connection...\n");

    while (idle());
		// connection ready
		if ((acc = accept(sock, NULL, NULL)) == -1) {printf("accept\n"); exit(1);}

		if (cur_pid) {
			// already handling a client, turn new one away
			c = 2; write(acc, &c, 1);  // write code 2 into socket (busy)
		}
		else {
			// handle connection
			c = 1; write(acc, &c, 1);  // write code 1 into socket (ok)

	    switch (cur_pid = fork()) {
				case -1: printf("fork: %i\n", errno); exit(1);
				case 0: accepthandler(acc); exit(0);
			}
		}
		close(acc); // no longer needed on overseer-side
	}
}
