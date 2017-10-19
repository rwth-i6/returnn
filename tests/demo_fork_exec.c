
#include <stdio.h>
#include <pthread.h>
// int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void));

static long magic_number = 0;

void set_magic_number(long i) {
    magic_number = i;
}

void hello_from_child() {
    printf("Hello from child atfork, magic number %li.\n", magic_number);
    fflush(stdout);
}

void register_hello_from_child() {
    pthread_atfork(0, 0, &hello_from_child);
}
