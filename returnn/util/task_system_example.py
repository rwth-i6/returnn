#!/usr/bin/env python

from __future__ import print_function

import sys
from .task_system import AsyncTask


def start():
  # Create a real subprocess, not just a fork.
  async_task = AsyncTask(func=process, name="My sub process", mustExec=True)

  # We have a connection (duplex pipe) to communicate.
  async_task.conn.send(print_action)
  assert async_task.conn.recv() == 42
  async_task.conn.send(sys.exit)

  async_task.join()


def print_action():
  print("Hello")
  return 42


def process(async_task):
  while True:
    action = async_task.conn.recv()
    res = action()
    async_task.conn.send(res)


if __name__ == "__main__":
  start()
