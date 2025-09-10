/*

232.用栈实现队列
力扣题目链接

使用栈实现队列的下列操作：

push(x) -- 将一个元素放入队列的尾部。
pop() -- 从队列首部移除元素。
peek() -- 返回队列首部的元素。
empty() -- 返回队列是否为空。

示例:

MyQueue queue = new MyQueue();
queue.push(1);
queue.push(2);
queue.peek();  // 返回 1
queue.pop();   // 返回 1
queue.empty(); // 返回 false
*/
class MyQueue {
  List<int> stack = [];
  List<int> tempStack = [];
  MyQueue();
  void push(int x) {
    stack.add(x);
  }

  int pop() {
    while (stack.isNotEmpty) {
      int temp = stack.removeLast();
      tempStack.add(temp);
    }
    int res = tempStack.removeLast();
    while (tempStack.isNotEmpty) {
      int temp = tempStack.removeLast();
      stack.add(temp);
    }
    return res;
  }

  int peek() {
    while (stack.isNotEmpty) {
      int temp = stack.removeLast();
      tempStack.add(temp);
    }
    int res = tempStack.last;
    while (tempStack.isNotEmpty) {
      int temp = tempStack.removeLast();
      stack.add(temp);
    }
    return res;
  }

  int empty() {
    return stack.isEmpty ? 1 : 0;
  }
}
