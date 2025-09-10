class MyStack {
  List<int> queue = [];
  MyStack() {}

  void push(int x) {
    queue.insert(0, x);
  }

  int pop() {
    int size = queue.length;
    while (size > 1) {
      int temp = queue.removeLast();
      queue.insert(0, temp);
      size--;
    }
    return queue.removeLast();
  }

  int top() {
    int size = queue.length;
    while (size > 1) {
      int temp = queue.removeLast();
      queue.insert(0, temp);
      size--;
    }
    int res = queue.last;
    queue.removeLast();
    queue.insert(0, res);
    return res;
  }

  bool empty() {
    return queue.isEmpty;
  }
}

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = MyStack();
 * obj.push(x);
 * int param2 = obj.pop();
 * int param3 = obj.top();
 * bool param4 = obj.empty();
 */