import 'dart:math';
import 'dart:collection';
import 'package:collection/collection.dart';

void main() {
  // findMedianSortedArrays([1, 4], [2]);
  // print(longestPalindrome("ababa"));
  // -8463847412
  // print(reverse(-2147483648));
  // print(myAtoi("   +042"));
  // print(longestCommonPrefix(["flower", "flow", ""]));
  // print(threeSum([0, 0, 0], 10000));
  // ListNode? head = ListNode(1, ListNode(2));
  // ListNode? result = removeNthFromEnd(head, 2);
  // while (result != null) {
  //   print(result.val);
  //   result = result.next;
  // }
  // print(generateParenthesis(3));
  // ListNode l1 = ListNode(
  //     -8,
  //     ListNode(-7,
  //         ListNode(-6, ListNode(-5, ListNode(-3, ListNode(-2, ListNode(0)))))));
  // ListNode l2 = ListNode(
  //     -9,
  //     ListNode(
  //         -5, ListNode(1, ListNode(2, ListNode(2, ListNode(4, ListNode(4)))))));
  // ListNode l3 =
  //     ListNode(-3, ListNode(-3, ListNode(-3, ListNode(-2, ListNode(2)))));
  // ListNode l4 = ListNode(
  //     -9,
  //     ListNode(-6,
  //         ListNode(-6, ListNode(-6, ListNode(-4, ListNode(-3, ListNode(2)))))));
  // ListNode l5 = ListNode(
  //     -8,
  //     ListNode(-7,
  //         ListNode(-3, ListNode(-2, ListNode(0, ListNode(1, ListNode(4)))))));
  // ListNode l6 = ListNode(-4, ListNode(0));
  // ListNode l7 =
  //     ListNode(-10, ListNode(-2, ListNode(-1, ListNode(1, ListNode(1)))));
  // ListNode l8 = ListNode(-10);
  // List<ListNode?> lists = [l1, l2, l3, l4, l5, l6, l7, l8];
  // ListNode? mergedList = mergeKLists(lists);
  // print(divide(10, 3));
  // print(combinationSum2([10, 1, 2, 7, 6, 1, 5], 8));
  // print(firstMissingPositive([3, 4, -1, 1]));
  // print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]));
  // print(jump([2, 3, 1, 1, 4]));
  // print(pow(2.5, 3));
  // print(merge([
  //   [1, 3],
  //   [2, 6],
  //   [8, 10],
  //   [15, 18]
  // ]));

  // print(uniquePathsWithObstacles([
  //   [1, 0]
  // ]));
  // print(minWindow("ADOBECODEBANC", 'ABC'));
  // print(search2([1], 2));
  // ListNode head = ListNode(1, ListNode(1, ListNode(2)));
  // print(deleteDuplicates2(head));
  // largestRectangleArea([2, 4]);
  // subsetsWithDup([1, 2, 2]);
  // [1,null,2,3]
  // TreeNode root = TreeNode(1, null, TreeNode(2, TreeNode(3)));
  // inOrder2(root);
  // [1,3,null,null,2]
  // TreeNode root = TreeNode(1, TreeNode(3, null, TreeNode(2)), null);
  // recoverTree(root);
  // [3,9,20,null,null,15,7]
  // TreeNode root =
  //     TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)));
  // levelOrder(root);
  // levelOrderBottom(root);
  // minimumTotal([
  //   [-1],
  //   [-2, -3]
  // ]);
  // longestConsecutive([1, 2, 3, 5]);
  // wordBreak2("catsanddog", ["cat", "cats", "and", "sand", "dog"]);
  // ListNode head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))));
  // reorderList(head);
  // convert("PAYPALISHIRING", 3);
  // List<int> result = spiralOrder([
  //   [1, 2, 3],
  //   [4, 5, 6],
  //   [7, 8, 9]
  // ]);
  // print(result);

  // 1,2,6,3,4,5,6
  // ListNode head = ListNode(
  //     1,
  //     ListNode(
  //         2, ListNode(6, ListNode(3, ListNode(4, ListNode(5, ListNode(6)))))));
  // ListNode? result = removeElements(head, 6);
  // while (result != null) {
  //   print(result.val);
  //   result = result.next;
  // }

  // isAnagram("ggii", "eekk");

  // intersect([4, 9, 5], [9, 4, 9, 8, 4]);
  // print(buildNextArray2("abcbabca"));
  // maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3);
  // [1,2,3,4,5,6,null,8]
  // isBalanced(TreeNode(
  //     1, TreeNode(2, TreeNode(3, TreeNode(4, TreeNode(5, TreeNode(6)))))));

// [5,4,8,11,null,13,4,7,2,null,null,null,1]
  // hasPathSum(
  //     TreeNode(
  //       5,
  //       TreeNode(
  //         4,
  //         TreeNode(
  //           11,
  //           TreeNode(7),
  //           TreeNode(2),
  //         ),
  //       ),
  //       TreeNode(
  //         8,
  //         TreeNode(13),
  //         TreeNode(
  //           4,
  //           null,
  //           TreeNode(1),
  //         ),
  //       ),
  //     ),
  //     22);

  // combinationSum3(9, 45);
  // letterCombinations('22222');
  // largestSumAfterKNegations([2, -3, -1, 5, -4], 2);
  // eraseOverlapIntervals([
  //   [-52, 31],
  //   [-73, -26],
  //   [82, 97],
  //   [-65, -11],
  //   [-62, -49],
  //   [95, 99],
  //   [58, 95],
  //   [-31, 49],
  //   [66, 98],
  //   [-63, 2],
  //   [30, 47],
  //   [-40, -26]
  // ]);
  longestPalindromeSubseq("cbbd");
}

/**
 * 1. 两数之和
 * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值 target 的那两个整数，并返回它们的数组下标。
 * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
 * 你可以按任意顺序返回答案。
 * 
 * 示例 1：
 * 输入：nums = [2,7,11,15], target = 9
 * 输出：[0,1]
 * 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
 */
List<int> twoSum(List<int> nums, int target) {
  Map<int, int> numToIndex = {};
  for (int i = 0; i < nums.length; i++) {
    int expectedNum = target - nums[i];
    if (numToIndex.containsKey(expectedNum)) {
      return [numToIndex[expectedNum]!, i];
    } else {
      numToIndex[nums[i]] = i;
    }
  }
  return [];
}

/**
 * 2. 两数相加
 * 给你两个非空的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。
 * 请你将两个数相加，并以相同形式返回一个表示和的链表。
 * 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
 * 
 * 示例 1：
 * 输入：l1 = [2,4,3], l2 = [5,6,4]
 * 输出：[7,0,8]
 * 解释：342 + 465 = 807。
 */
ListNode? addTwoNumbers(ListNode? l1, ListNode? l2) {
  ListNode dummy = ListNode(0);
  ListNode current = dummy;
  int upTen = 0;
  while (l1 != null || l2 != null || upTen > 0) {
    int val1 = l1?.val ?? 0;
    int val2 = l2?.val ?? 0;

    int sum = (val1 + val2 + upTen) % 10;
    upTen = (val1 + val2 + upTen) ~/ 10;
    current.next = ListNode(sum);
    current = current.next!;
    l1 = l1?.next;
    l2 = l2?.next;
  }
  return dummy.next;
}

/**
 * 3. 无重复字符的最长子串
 * 给定一个字符串 s ，请你找出其中不含有重复字符的最长子串的长度。
 * 
 * 示例 1：
 * 输入: s = "abcabcbb"
 * 输出: 3 
 * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
 */
int lengthOfLongestSubstring(String s) {
  int slow = 0;
  int maxLength = 0;
  Set<String> sSet = {};
  for (int fast = 0; fast < s.length; fast++) {
    while (sSet.contains(s[fast])) {
      sSet.remove(s[slow]);
      slow++;
    }
    sSet.add(s[fast]);
    maxLength = maxLength > (fast - slow + 1) ? maxLength : (fast - slow + 1);
  }
  return maxLength;
}

/**
 * 4. 寻找两个正序数组的中位数
 * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的中位数。
 * 算法的时间复杂度应该为 O(log (m+n)) 。
 * 
 * 示例 1：
 * 输入：nums1 = [1,3], nums2 = [2]
 * 输出：2.00000
 * 解释：合并数组 = [1,2,3]，中位数 2
 * 
 * 示例 2：
 * 输入：nums1 = [1,2], nums2 = [3,4]
 * 输出：2.50000
 * 解释：合并数组 = [1,2,3,4]，中位数 (2 + 3) / 2 = 2.5
 */
double findMedianSortedArrays(List<int> nums1, List<int> nums2) {
  int totalLength = nums1.length + nums2.length;
  int midIndex = totalLength ~/ 2;
  bool isEven = totalLength % 2 == 0;
  int i = 0, j = 0;
  List<int> merged = [];
  print(totalLength);
  print(midIndex);
  print(isEven);

  while (i + j <= midIndex) {
    if (i < nums1.length && nums1[i] < nums2[j] || j == nums2.length) {
      merged.add(nums1[i]);
      i++;
    } else {
      merged.add(nums2[j]);
      j++;
    }
  }
  print(i);
  print(j);
  print('object');
  for (int k = 0; k <= midIndex; k++) {
    print(merged[k]);
  }
  // return 1.0;
  if (isEven) {
    double midValue = (merged[midIndex - 1] + merged[midIndex]) / 2.0;
    print(midValue);
    return midValue;
  } else {
    double midValue = merged[midIndex].toDouble();
    print(midValue);
    return midValue;
  }
}

/**
 * 5. 最长回文子串
 * 给你一个字符串 s，找到 s 中最长的回文子串。
 * 
 * 示例 1：
 * 输入：s = "babad"
 * 输出："bab"
 * 解释："aba" 同样是符合题意的答案。
 */
String longestPalindrome(String s) {
  String longest = s.substring(0, 1);
  int maxLen = 1;

  for (int i = 1; i < s.length; i++) {
    int left = i;
    int right = i;
    while (left >= 0 && right <= s.length - 1) {
      if (s[left] == s[right]) {
        if (maxLen < right - left + 1) {
          maxLen = right - left + 1;
          longest = s.substring(left, right + 1);
        }
        left--;
        right++;
      } else {
        break;
      }
    }
    if (s[i - 1] == s[i]) {
      left = i - 1;
      right = i;
      while (left >= 0 && right <= s.length - 1) {
        if (s[left] == s[right]) {
          if (maxLen < right - left + 1) {
            maxLen = right - left + 1;
            longest = s.substring(left, right + 1);
          }
          left--;
          right++;
        } else {
          break;
        }
      }
    }
  }
  return longest;
}

/**
 * 6. Z 字形变换
 * 将一个给定字符串 s 根据给定的行数 numRows，以从上往下、从左到右进行 Z 字形排列。
 * 之后，你的输出需要从左往右逐行读取，产生出一个新的字符串。
 * 
 * 示例 1：
 * 输入：s = "PAYPALISHIRING", numRows = 3
 * 输出："PAHNAPLSIIGYIR"
 * 解释：
 * P   A   H   N
 * A P L S I I G
 * Y   I   R
 */
// "PAYPALISHIRING"

String convert(String s, int numRows) {
  if (numRows <= 1) return s; // if only one row, return the string as is
  List<String> result = List.filled(numRows, '');
  bool goDown = false; // flag to indicate direction
  int currentRow = 0; // start from the first row
  for (int i = 0; i < s.length; i++) {
    result[currentRow] += s[i]; // append the character to the current row
    if (currentRow == 0 || currentRow == numRows - 1) {
      goDown = !goDown;
    }
    if (goDown) {
      currentRow++; // move down
    } else {
      currentRow--; // move up
    }
  }

  return result.join(''); // return the final string
}

/**
 * 7. 整数反转
 * 给你一个 32 位的有符号整数 x，返回将 x 中的数字部分反转后的结果。
 * 如果反转后整数超过 32 位的有符号整数的范围 [−2³¹, 2³¹ − 1]，就返回 0。
 * 假设环境不允许存储 64 位整数（有符号或无符号）。
 * 
 * 示例 1：
 * 输入：x = 123
 * 输出：321
 */
int reverse(int x) {
  // −231,  231 − 1
  int result = 0;
  int flag = x < 0 ? -1 : 1;
  x = x.abs();
  int MAX = 2147483647;
  int MIN = -2147483648;

  while (x != 0) {
    if (flag == -1 && result == MAX ~/ 10 && x % 10 == 8) {
      return MIN; // special case for -2147483648
    }
    if (result > MAX ~/ 10 || result == MAX ~/ 10 && x % 10 > 7) {
      return 0; // overflow
    }
    result = result * 10 + x % 10;
    x = x ~/ 10;
  }
  return result * flag;
}

/**
 * 8. 字符串转换整数 (atoi)
 * 请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数。
 * 函数 myAtoi(string s) 的算法如下：
 * 1. 读入字符串并丢弃无用的前导空格
 * 2. 检查下一个字符为 '-' 还是 '+'（如果两者都不存在，则假定结果为正）
 * 3. 读取数字字符，直到遇到非数字字符或到达字符串的结尾
 * 4. 如果整数超过 32 位有符号整数范围 [−2³¹, 2³¹ − 1]，需要截断
 * 
 * 示例 1：
 * 输入：s = "42"
 * 输出：42
 */
int myAtoi(String s) {
  s = s.trim();
  int flag = s.startsWith(RegExp(r'[-+]')) ? (s.startsWith('-') ? -1 : 1) : 1;
  int result = 0;
  print(s);
  int i = s.startsWith(RegExp(r'[-+]')) ? 1 : 0; // start after sign if present
  for (i; i < s.length; i++) {
    if (s.codeUnitAt(i) < '0'.codeUnitAt(0) ||
        s.codeUnitAt(i) > '9'.codeUnitAt(0)) {
      break; // stop at first non-digit character
    }
    result = result * 10 + s.codeUnitAt(i) - '0'.codeUnitAt(0);
    if (result > 2147483647) {
      return flag == 1 ? 2147483647 : -2147483648; // handle overflow
    }
  }
  return result * flag;
}

/**
 * 9. 回文数
 * 给你一个整数 x，如果 x 是一个回文整数，返回 true；否则返回 false。
 * 回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
 * 
 * 示例 1：
 * 输入：x = 121
 * 输出：true
 * 
 * 示例 2：
 * 输入：x = -121
 * 输出：false
 * 解释：从左向右读为 "-121"，从右向左读为 "121-"，因此不是回文数。
 */
bool isPalindrome(int x) {
  String s = x.toString();
  int left = 0;
  int right = s.length - 1;
  while (left < right) {
    if (s[left] != s[right]) {
      return false;
    }
    left++;
    right--;
  }
  return true;
}

/**
 * 11. 盛最多水的容器
 * 给定一个长度为 n 的整数数组 height。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i])。
 * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
 * 返回容器可以储存的最大水量。
 * 
 * 示例 1：
 * 输入：[1,8,6,2,5,4,8,3,7]
 * 输出：49 
 * 解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水的最大值为 49。
 */
int maxArea(List<int> height) {
// [1,8,6,2,5,4,8,3,7]
// 输出：49
  int left = 0;
  int right = height.length - 1;
  int sum = 0;
  while (left < right) {
    sum = max(sum, min(height[left], height[right]) * (right - left));
    if (height[left] < height[right]) {
      left++;
    } else {
      right--;
    }
  }

  return sum;
}

/**
 * 14. 最长公共前缀
 * 编写一个函数来查找字符串数组中的最长公共前缀。
 * 如果不存在公共前缀，返回空字符串 ""。
 * 
 * 示例 1：
 * 输入：strs = ["flower","flow","flight"]
 * 输出："fl"
 * 
 * 示例 2：
 * 输入：strs = ["dog","racecar","car"]
 * 输出：""
 * 解释：输入不存在公共前缀。
 */
String longestCommonPrefix(List<String> strs) {
  String result = '';
  if (strs.length == 0) return result;
  String first = strs[0];
  for (int i = 0; i < first.length; i++) {
    for (int j = 1; j < strs.length; j++) {
      if (i > strs[j].length - 1 || first[i] != strs[j][i]) {
        return result;
      }
    }
    result += first[i];
  }
  return result;
}

/**
 * 15. 三数之和
 * 给你一个整数数组 nums，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k，
 * 同时还满足 nums[i] + nums[j] + nums[k] == 0。请你返回所有和为 0 且不重复的三元组。
 * 注意：答案中不可以包含重复的三元组。
 * 
 * 示例 1：
 * 输入：nums = [-1,0,1,2,-1,-4]
 * 输出：[[-1,-1,2],[-1,0,1]]
 * 解释：
 * nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0。
 * nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0。
 * nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0。
 * 不同的三元组是 [-1,0,1] 和 [-1,-1,2]。
 */
int threeSum(List<int> nums, int target) {
  //思路：先排序，然后固定一个数，双指针夹逼
  nums.sort();
  int minDiff = 100001; // Initialize to a large value
  int left = 0;
  int right = nums.length - 1;
  for (int i = 0; i < nums.length; i++) {
    left = i + 1;
    right = nums.length - 1;
    if (i > 1 && nums[i] == nums[i - 1]) {
      continue; // skip duplicates
    }
    while (left < right) {
      int sum = nums[i] + nums[left] + nums[right];
      int diff = sum - target;
      if (diff.abs() < minDiff.abs()) {
        minDiff = diff;
      }
      if (sum == target) {
        return target;
      } else if (sum < target) {
        left++;
      } else {
        right--;
      }
    }
  }
  return minDiff + target;
}

/**
 * 16. 最接近的三数之和
 * 给你一个长度为 n 的整数数组 nums 和一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。
 * 返回这三个数的和。
 * 假定每组输入只存在恰好一个解。
 * 
 * 示例 1：
 * 输入：nums = [-1,2,1,-4], target = 1
 * 输出：2
 * 解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2)。
 * 
 * 示例 2：
 * 输入：nums = [0,0,0], target = 1
 * 输出：0
 * 解释：与 target 最接近的和是 0（0 + 0 + 0 = 0）。
 * 3 <= nums.length <= 1000
-1000 <= nums[i] <= 1000
-104 <= target <= 104

 */
int threeSumClosest(List<int> nums, int target) {
  nums.sort();
  print(nums);
  int minDiff = 100001;
  int left = 0;
  int right = nums.length - 1;
  for (int i = 0; i < nums.length; i++) {
    left = i + 1;
    right = nums.length - 1;
    if (i > 1 && nums[i] == nums[i - 1]) {
      continue; // skip duplicates
    }
    while (left < right) {
      int sum = nums[i] + nums[left] + nums[right];
      int diff = sum - target;
      if (diff.abs() < minDiff.abs()) {
        minDiff = diff;
      }
      if (sum == target) {
        return target;
      } else if (sum < target) {
        left++;
      } else {
        right--;
      }
    }
  }
  return minDiff + target;
}

/**
 * 17. 电话号码的字母组合
 * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按任意顺序返回。
 * 数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
 * 
 * 示例 1：
 * 输入：digits = "23"
 * 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
 */
List<String> letterCombinations(String digits) {
  if (digits.isEmpty) return [];
  List<String> result = [];
  List<String> mapping = [
    "",
    "",
    "abc",
    "def",
    "ghi",
    "jkl",
    "mno",
    "pqrs",
    "tuv",
    "wxyz"
  ];
  void backTrack(int index, String current) {
    if (current.length == digits.length) {
      result.add(current);
      return;
    }
    String letter = mapping[int.parse(digits[index])];
    for (int i = 0; i < letter.length; i++) {
      current += letter[i];
      backTrack(index + 1, current); // 注意这里是index+1不是i+1
      current = current.substring(0, current.length - 1); // backtrack
    }
  }

  backTrack(0, "");
  return result;
}

/**
 * 18. 四数之和
 * 给你一个由 n 个整数组成的数组 nums，和一个目标值 target。
 * 请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]]：
 * 0 <= a, b, c, d < n
 * a、b、c 和 d 互不相同
 * nums[a] + nums[b] + nums[c] + nums[d] == target
 * 
 * 示例 1：
 * 输入：nums = [1,0,-1,0,-2,2], target = 0
 * 输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
 */
List<List<int>> fourSum(List<int> nums, int target) {
  //思路同三数之和，只是需要加多一层for循环
  List<List<int>> result = [];
  if (nums.length < 4) return [];
  nums.sort();
  for (int i = 0; i < nums.length; i++) {
    if (i > 0 && nums[i] == nums[i - 1]) continue; // skip duplicates
    for (int j = i + 1; j < nums.length; j++) {
      if (nums[j] == nums[j - 1] && j > i + 1) continue; // skip duplicates
      int left = j + 1;
      int right = nums.length - 1;

      while (left < right) {
        int sum = nums[i] + nums[j] + nums[left] + nums[right];
        if (sum == target) {
          result.add([nums[i], nums[j], nums[left], nums[right]]);
          while (left < right && nums[left] == nums[left + 1]) {
            left++; // skip duplicates
          }
          while (right > left && nums[right] == nums[right - 1]) {
            right--; // skip duplicates
          }
          left++;
          right--;
        } else if (sum < target) {
          left++;
        } else {
          right--;
        }
      }
    }
  }
  return result;
}

/**
 454. 四数相加 II
给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。

为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -2^28 到 2^28 - 1 之间，最终结果不会超过 2^31 - 1 。
输入:

A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]
输出:

2

解释:

两个元组如下:

(0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
(1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
 */
int fourSumCount(
    List<int> nums1, List<int> nums2, List<int> nums3, List<int> nums4) {
  //思路：先把nums1和nums2的所有组合存到map中，然后遍历nums3和nums4，找map中是否有对应的值
  int result = 0;
  Map<int, int> sumMap = {};
  for (int i in nums1) {
    for (int j in nums2) {
      int sum = i + j;
      sumMap[sum] = (sumMap[sum] ?? 0) + 1;
    }
  }
  for (int i in nums3) {
    for (int j in nums4) {
      int expectedSum = -(i + j);
      if (sumMap.containsKey(expectedSum)) {
        // expectedSum可能是有多种组合方式的
        result += sumMap[expectedSum]!;
      }
    }
  }
  return result;
}

/**
 344. 反转字符串
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

输入：s = ["h","e","l","l","o"]
输出：["o","l","l","e","h"]
示例 2：

输入：s = ["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]
 */
void reverseString(List<String> s) {
  int left = 0;
  int right = s.length - 1;
  while (left < right) {
    String temp = s[left];
    s[left] = s[right];
    s[right] = temp;
    left++;
    right--;
  }
}

/*
151. 反转字符串中的单词
给你一个字符串 s ，请你反转字符串中 单词 的顺序。

单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。

返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。

注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
输入：s = "the sky is blue"
输出："blue is sky the"
示例 2：

输入：s = "  hello world  "
输出："world hello"
解释：反转后的字符串中不能存在前导空格和尾随空格。
示例 3：

输入：s = "a good   example"
输出："example good a"
解释：如果两个单词间有多余的空格，反转后的字符串需要将单词间的空格减少到仅有一个。
  */
String reverseWords(String s) {
  String newS = s.trim();
  List<String> words = newS.split(RegExp(r'\s+'));
  for (int i = 0; i < words.length; i++) {
    int left = 0;
    int right = words[i].length - 1;
    while (left < right) {
      String temp = words[i][left];
      words[i] = words[i].replaceRange(left, left + 1, words[i][right]);
      words[i] = words[i].replaceRange(right, right + 1, temp);
      left++;
      right--;
    }
  }
  newS = words.join(' ');
  int left = 0;
  int right = newS.length - 1;
  while (left < right) {
    String temp = newS[left];
    newS = newS.replaceRange(left, left + 1, newS[right]);
    newS = newS.replaceRange(right, right + 1, temp);
    left++;
    right--;
  }
  return newS;
}

class ListNode {
  int val;
  ListNode? next;
  ListNode([this.val = 0, this.next]);
}

class TreeNode {
  int val;
  TreeNode? left;
  TreeNode? right;
  TreeNode([this.val = 0, this.left, this.right]);
}

// 删除链表中倒数第 n 个节点
ListNode? removeNthFromEnd(ListNode? head, int n) {
  if (head == null || n <= 0) return head;
  ListNode? dummy = ListNode(0, head);
  ListNode? slow = dummy;
  ListNode? fast = dummy;
  for (int i = 0; i < n; i++) {
    print(fast?.val);
    fast = fast?.next;
  }
  while (fast?.next != null) {
    slow = slow?.next;
    fast = fast?.next;
  }
  if (slow?.next == null)
    return null; // if n is equal to the length of the list
  slow?.next = slow.next?.next; // remove the nth node from end
  return dummy.next;
}

/**

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。

图示两个链表在节点 c1 开始相交：
 */
ListNode? getIntersectionNode(ListNode? headA, ListNode? headB) {
  if (headA == null || headB == null) return null;
  ListNode? a = headA;
  ListNode? b = headB;
  int aSize = 0;
  int bSize = 0;
  while (a != null) {
    aSize++;
    a = a.next;
  }
  while (b != null) {
    bSize++;
    b = b.next;
  }
  a = headA;
  b = headB;
  if (aSize > bSize) {
    for (int i = 0; i < aSize - bSize; i++) {
      a = a?.next;
    }
  } else {
    for (int i = 0; i < bSize - aSize; i++) {
      b = b?.next;
    }
  }
  while (a != null && b != null) {
    if (a == b) {
      return a;
    }
    a = a.next;
    b = b.next;
  }
  return a;
}

// 找到链表环的起始节点
ListNode? detectCycle(ListNode? head) {
  // 思路：先找到相遇点，然后从头节点和相遇点同时出发，每次走一步，相遇点即为环的起始节点
  ListNode? slow = head;
  ListNode? fast = head;
  while (fast != null && fast.next != null) {
    slow = slow?.next;
    fast = fast.next?.next;
    if (slow == fast) {
      ListNode? ptr = head;
      while (ptr != slow) {
        ptr = ptr?.next;
        slow = slow?.next;
      }
      return ptr;
    }
  }
  // 无环
  return null;
}

/**
 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

示例 1: 输入: s = "anagram", t = "nagaram" 输出: true

示例 2: 输入: s = "rat", t = "car" 输出: false

说明: 你可以假设字符串只包含小写字母。
 */
bool isAnagram(String s, String t) {
  if (s.length != t.length) return false;
  List<int> count = List.filled(26, 0);
  for (int i = 0; i < t.length; i++) {
    count[s.codeUnits[i] - 'a'.codeUnitAt(0)]++;
    count[t.codeUnits[i] - 'a'.codeUnitAt(0)]--;
  }
  for (int i = 0; i < 26; i++) {
    if (count[i] > 0) return false;
  }
  return true;
}

/**
 383. 赎金信
给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。
(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)
注意：

你可以假设两个字符串均只含有小写字母。
 */

bool canConstruct(String ransomNote, String magazine) {
  List<int> count = List.filled(26, 0);
  for (int i = 0; i < ransomNote.length; i++) {
    count[ransomNote.codeUnits[i] - 'a'.codeUnitAt(0)]++;
  }
  for (int i = 0; i < magazine.length; i++) {
    count[magazine.codeUnits[i] - 'a'.codeUnitAt(0)]--;
  }
  for (int i = 0; i < 26; i++) {
    if (count[i] > 0) return false;
  }
  return true;
}

/*
20. 有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。
输入：s = "()"

输出：true
输入：s = "()[]{}"

输出：true
*/
bool isValid(String s) {
  List<String> stack = [];
  for (int i = 0; i < s.length; i++) {
    String current = s[i];
    if (current == '(' || current == '{' || current == '[') {
      stack.add(current);
    } else if (current == ')') {
      if (stack.length == 0) return false;
      String pop = stack.removeLast();
      if (pop != '(') {
        return false;
      }
    } else if (current == ']') {
      if (stack.length == 0) return false;
      String pop = stack.removeLast();
      if (pop != '[') {
        return false;
      }
    } else if (current == '}') {
      if (stack.length == 0) return false;
      String pop = stack.removeLast();
      if (pop != '{') {
        return false;
      }
    }
  }
  return stack.length == 0;
}

/**
 1047. 删除字符串中的所有相邻重复项
给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

输入："abbaca"
输出："ca"
 */

String removeDuplicates1047(String s) {
  List<String> stack = [];
  String result = '';
  for (int i = 0; i < s.length; i++) {
    if (stack.isNotEmpty) {
      if (stack.last == s[i]) {
        stack.removeLast();
      } else {
        stack.add(s[i]);
      }
    } else {
      stack.add(s[i]);
    }
  }
  while (stack.isNotEmpty) {
    result = stack.removeLast() + result;
  }
  return result;
}

ListNode? mergeTwoLists(ListNode? list1, ListNode? list2) {
  if (list1 == null) return list2;
  if (list2 == null) return list1;
  ListNode? dummy = ListNode(0, list1);
  ListNode current = dummy;
  while (list1 != null && list2 != null) {
    if (list1.val < list2.val) {
      current.next = list1;
      list1 = list1.next;
    } else {
      current.next = list2;
      list2 = list2.next;
    }
    current = current.next!;
  }
  current.next = list1 ?? list2; // append the remaining nodes
  return dummy.next;
}

List<String> generateParenthesis(int n) {
  int left = 0;
  int right = 0;

  List<String> result = [];
  void backTrack(int index, String current) {
    if (left > n || right > n || right > left) {
      return;
    }
    if (current.length == n * 2) {
      result.add(current);
      return;
    }
    for (int i = index; i < 2 * n; i++) {
      current += '(';
      left++;
      backTrack(i + 1, current);
      current = current.substring(0, current.length - 1); // backtrack
      left--;

      current += ')';
      right++;
      backTrack(i + 1, current);
      current = current.substring(0, current.length - 1); // backtrack
      right--;
    }
  }

  backTrack(0, "");
  return result;
}

ListNode? mergeKLists(List<ListNode?> lists) {
  List<ListNode> mergedLists = [];
  for (ListNode? list in lists) {
    while (list != null) {
      mergedLists.add(list);
      list = list.next;
    }
  }
  mergedLists.sort((ListNode a, ListNode b) => a.val.compareTo(b.val));
  if (mergedLists.isEmpty) return null;
  ListNode? dummy = ListNode(0, mergedLists[0]);
  for (int i = 0; i < mergedLists.length - 1; i++) {
    mergedLists[i].next = mergedLists[i + 1];
  }
  return dummy.next;
}

ListNode? swapPairs(ListNode? head) {
  if (head == null || head.next == null) return head;
  ListNode dummy = ListNode(0, head);
  ListNode? pre = dummy;
  ListNode? current = dummy.next!;
  ListNode? last = dummy.next!.next!;
  while (last != null) {
    // ListNode? nextPair = last.next;

    // pre?.next = last;
    // current?.next = last.next;
    // last.next = current;

    // pre = current;
    // current = nextPair;
    // last = nextPair != null ? nextPair.next : null;
    ListNode? nextPair = last.next;

    pre?.next = last;
    last.next = current;
    current?.next = nextPair;

    pre = current;
    current = nextPair;
    last = nextPair != null ? nextPair.next : null;
  }
  return dummy.next;
}

int removeDuplicates(List<int> nums) {
  if (nums.length <= 1) return nums.length;
  int slow = 1;
  for (int fast = 1; fast < nums.length; fast++) {
    if (nums[slow - 1] != nums[fast]) {
      nums[slow] = nums[fast];
      slow++;
    }
  }
  return slow;
}

int removeElement(List<int> nums, int val) {
  int slow = 0;
  for (int fast = 0; fast < nums.length; fast++) {
    if (nums[fast] != val) {
      nums[slow] = nums[fast];
      slow++;
    }
  }
  return slow + 1;
}

// 注意：假设我们的环境只能存储 32 位 有符号整数，其数值范围是 [−231,  231 − 1] 。本题中，
// 如果商 严格大于 231 − 1 ，则返回 231 − 1 ；如果商 严格小于 -231 ，则返回 -231 。
int divide(int dividend, int divisor) {
  if (divisor == 0) return 0;
  if (divisor == 1) return dividend;
  if (divisor == -1) return dividend == -2147483648 ? 2147483647 : -dividend;
  int result = 0;
  int flag = 1;
  if (dividend < 0 && divisor < 0 || dividend > 0 && divisor > 0) {
    flag = 1;
  } else {
    flag = -1;
  }
  dividend = dividend.abs();
  divisor = divisor.abs();
  int addDivisor = divisor;
  int sumDivisor = 1;
  while (dividend >= divisor) {
    dividend -= addDivisor;
    result += sumDivisor;
    addDivisor += addDivisor;
    sumDivisor += sumDivisor;
    while (dividend < addDivisor && dividend >= divisor) {
      addDivisor = divisor;
      sumDivisor = 1;
    }
  }
  return flag > 0 ? result : -result;
}

void nextPermutation(List<int> nums) {
  if (nums.length <= 1) return;
  int i, k;
  for (i = nums.length - 2; i >= 0; i--) {
    if (nums[i] < nums[i + 1]) {
      break;
    }
  }
  if (i < 0) {
    nums.sort();
    return;
  }
  for (k = nums.length - 1; k > i; k--) {
    if (nums[k] > nums[i]) {
      int temp = nums[i];
      nums[i] = nums[k];
      nums[k] = temp;
      break;
    }
  }
  int left = i + 1;
  int right = nums.length - 1;
  while (left < right) {
    int temp = nums[left];
    nums[left] = nums[right];
    nums[right] = temp;
    left++;
    right--;
  }
}

int longestValidParentheses(String s) {
  List<int> dp = List.filled(s.length, 0);
  int maxLength = 0;
  for (int i = 0; i < s.length; i++) {
    if (s[i] == ')') {
      if (i > 0 && s[i - 1] == '(') {
        if (i - 2 >= 0) {
          dp[i] = dp[i - 2] + 2;
        } else {
          dp[i] = 2;
        }
      } else if (i > 0 && s[i - 1] == ')') {
        dp[i] =
            dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
      }
    }
    maxLength = max(maxLength, dp[i]);
  }
  return maxLength;
}

// 4,5,6,7,0,1,2,3
int search(List<int> nums, int target) {
  int midIndex = nums.length ~/ 2;
  int left = 0;
  int right = nums.length - 1;
  while (left <= right) {
    midIndex = (left + right) ~/ 2;
    if (nums[midIndex] == target) {
      return midIndex;
    }
    if (nums[0] <= nums[midIndex]) {
      // left part is sorted
      if (nums[left] <= target && target < nums[midIndex]) {
        right = midIndex - 1; // search in left part
      } else {
        left = midIndex + 1; // search in right part
      }
    } else {
      // right part is sorted
      if (nums[midIndex] < target && target <= nums[right]) {
        left = midIndex + 1; // search in right part
      } else {
        right = midIndex - 1; // search in left part
      }
    }
  }
  return -1;
}

// 0,1,2,3,4,5,6,7
List<int> searchRange(List<int> nums, int target) {
  List<int> search(List<int> nums, double target) {
    int left = 0;
    int right = nums.length - 1;
    while (left <= right) {
      int mid = (left + right) ~/ 2;
      if (target < nums[mid]) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return [left, right];
  }

  double targetLeft = target - 0.5;
  double targetRight = target + 0.5;
  int resultLeft = search(nums, targetLeft)[0];
  int resultRight = search(nums, targetRight)[1];
  if (resultLeft > resultRight) {
    return [-1, -1];
  } else {
    return [resultLeft, resultRight];
  }
}

int searchInsert(List<int> nums, int target) {
  if (nums.isEmpty) return 0;
  int left = 0;
  int right = nums.length - 1;
  while (left <= right) {
    int mid = (left + right) ~/ 2;
    if (nums[mid] == target) {
      return mid;
    } else if (nums[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return left;
}

bool isValidSudoku(List<List<String>> board) {
  List<List<int>> rows = List.generate(9, (_) => List.filled(9, 0));
  List<List<int>> cols = List.generate(9, (_) => List.filled(9, 0));
  List<List<int>> boxes = List.generate(9, (_) => List.filled(9, 0));

  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      String s = board[i][j];
      if (s == '.') continue;
      int index = s.codeUnitAt(0) - '1'.codeUnitAt(0);
      rows[i][index]++;
      cols[j][index]++;
      boxes[(i ~/ 3) * 3 + (j ~/ 3)][index]++;
      if (rows[i][index] > 1 ||
          cols[j][index] > 1 ||
          boxes[(i ~/ 3) * 3 + (j ~/ 3)][index] > 1) {
        return false; // duplicate found
      }
    }
  }
  return true;
}

/**
 37. 解数独
编写一个程序，通过填充空格来解决数独问题。

一个数独的解法需遵循如下规则： 数字 1-9 在每一行只能出现一次。 数字 1-9 在每一列只能出现一次。 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。 空白格用 '.' 表示。
 */
void solveSudoku(List<List<String>> board) {
  List<List<int>> rows = List.generate(9, (_) => List.filled(9, 0));
  List<List<int>> cols = List.generate(9, (_) => List.filled(9, 0));
  List<List<int>> boxes = List.generate(9, (_) => List.filled(9, 0));
  // Helper function to check if placing k at (i, j) is valid
  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      String s = board[i][j];
      if (s == '.') continue;
      int index = s.codeUnitAt(0) - '1'.codeUnitAt(0);
      rows[i][index]++;
      cols[j][index]++;
      boxes[(i ~/ 3) * 3 + (j ~/ 3)][index]++;
    }
  }
  bool isValid(int i, int j, int k) {
    rows[i][k - 1]++;
    cols[j][k - 1]++;
    boxes[(i ~/ 3) * 3 + (j ~/ 3)][k - 1]++;
    bool result = rows[i][k - 1] <= 1 &&
        cols[j][k - 1] <= 1 &&
        boxes[(i ~/ 3) * 3 + (j ~/ 3)][k - 1] <= 1;
    if (!result) {
      rows[i][k - 1]--;
      cols[j][k - 1]--;
      boxes[(i ~/ 3) * 3 + (j ~/ 3)][k - 1]--;
    }
    return result;
  }

  bool backTrack() {
    for (int i = 0; i < 9; i++) {
      for (int j = 0; j < 9; j++) {
        String s = board[i][j];
        if (s == '.') {
          for (int k = 1; k <= 9; k++) {
            if (isValid(i, j, k)) {
              board[i][j] = k.toString();
              if (backTrack()) return true;
              board[i][j] = '.'; // backtrack

              rows[i][k - 1]--;
              cols[j][k - 1]--;
              boxes[(i ~/ 3) * 3 + (j ~/ 3)][k - 1]--;
            }
          }
          return false;
        }
      }
    }
    return true; // all cells filled
  }

  backTrack();
}

/**
51. N皇后
n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
 */
List<List<String>> solveNQueens(int n) {
  List<List<String>> result = [];
  List<List<String>> current = List.generate(n, (_) => List.filled(n, '.'));
  bool isValid(int row, int col, List<List<String>> current) {
    for (int i = 0; i < row; i++) {
      if (current[i][col] == 'Q') return false; // check column
    }
    for (int i = 0; i < col; i++) {
      if (current[row][i] == 'Q') return false; // check row
    }
    for (int i = row - 1, j = col + 1; i >= 0 && j <= n - 1; i--, j++) {
      if (current[i][j] == 'Q') return false; // check right diagonal
    }
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
      if (current[i][j] == 'Q') return false; // check left diagonal
    }
    return true;
  }

  List<String> listToString(List<List<String>> current) {
    List<String> newCurrent = [];
    for (int i = 0; i < n; i++) {
      String rowString = current[i].join('');
      newCurrent.add(rowString);
    }
    return newCurrent;
  }

  void backTrack(int row) {
    if (row == n) {
      result.add(List.from(listToString(current)));
      return;
    }

    for (int col = 0; col < n; col++) {
      if (isValid(row, col, current)) {
        current[row][col] = 'Q';
        backTrack(row + 1);
        current[row][col] = '.'; // backtrack
      }
    }
  }

  backTrack(0);
  return result;
}

// 123445
String countAndSay(int n) {
  String result = '1';
  for (int i = 1; i < n; i++) {
    String temp = result[0];
    int count = 1;
    String newResult = '';
    for (int j = 1; j < result.length; j++) {
      if (result[j] == temp) {
        count++;
      } else {
        newResult += '$count$temp';
        temp = result[j];
        count = 1;
      }
    }
    result = newResult + '$count$temp'; // append last counted group
  }
  return result;
}

/**
  39. 组合总和
给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。
说明：
所有数字（包括 target）都是正整数。
解集不能包含重复的组合。
示例 1：

输入：candidates = [2,3,6,7], target = 7,
所求解集为： [ [7], [2,2,3] ]
示例 2：

输入：candidates = [2,3,5], target = 8,
所求解集为： [ [2,2,2,2], [2,3,3], [3,5] ]
 */
List<List<int>> combinationSum(List<int> candidates, int target) {
  List<List<int>> result = [];
  List<int> current = [];
  int sum = 0;
  void backTrack(int index) {
    if (sum == target) {
      result.add(List.from(current));
      // make a copy of current list
      return;
    }
    if (sum > target) {
      return;
    }
    for (int i = index; i < candidates.length; i++) {
      current.add(candidates[i]);
      sum += candidates[i];
      backTrack(i); // allow same element to be used again
      current.removeLast();
      sum -= candidates[i]; // backtrack
    }
  }

  backTrack(0);
  return result;
}

/**
 40.组合总和II
给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

说明： 所有数字（包括目标数）都是正整数。解集不能包含重复的组合。 

示例 1:
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
示例 2:
输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]
 */
// 1 2 2 2 ， 5
List<List<int>> combinationSum2(List<int> candidates, int target) {
  List<List<int>> result = [];
  List<int> current = [];
  int sum = 0;
  List<bool> isUsed = List.filled(candidates.length, false);
  candidates.sort();
  void backTrack(int index) {
    if (sum == target) {
      result.add(List.from(current));
      return;
    }
    if (sum > target) {
      return;
    }
    for (int i = index; i < candidates.length; i++) {
      // if (i > 0 && candidates[i] == candidates[i - 1] && !isUsed[i - 1]) {
      //   continue; // 1 2 2 2, 第一个2是当前树枝上个节点正在用，第二个2是当前树层使用过的，第三个2是需要跳过的
      // }
      if (i > index && candidates[i] == candidates[i - 1]) {
        continue; // // 1 2 2 2, i>index直接不需要考虑当前树枝之上使用过的2，第二个2是当前树层使用过的，第三个2是需要跳过的
      }
      current.add(candidates[i]);
      sum += candidates[i];
      // isUsed[i] = true; // mark as used
      backTrack(i + 1); // allow same element to be used again
      current.removeLast();
      sum -= candidates[i]; // backtrack
      // isUsed[i] = false; // mark as unused
    }
  }

  backTrack(0);
  return result;
}

int firstMissingPositive(List<int> nums) {
  int n = nums.length;
  for (int i = 0; i < n; i++) {
    while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
      int temp = nums[nums[i] - 1];
      nums[nums[i] - 1] = nums[i];
      nums[i] = temp;
    }
  }
  for (int i = 0; i < n; i++) {
    if (nums[i] != i + 1) {
      return i + 1;
    }
  }
  return n + 1;
}

/**
 42. 接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

输入：height = [4,2,0,3,2,5]
输出：9
 */
// [0,1,0,2,1,0,1,3,2,1,2,1]
int trap(List<int> height) {
  // 双指针法，用两个数组记录左右两边的最大值
  // List<int> dpLeft = List.filled(height.length, 0);
  // List<int> dpRight = List.filled(height.length, 0);
  // int result = 0;
  // for (int i = 0; i < height.length; i++) {
  //   if (i == 0) {
  //     dpLeft[i] = height[i];
  //   } else {
  //     dpLeft[i] = max(dpLeft[i - 1], height[i]);
  //   }
  // }
  // for (int i = height.length - 1; i >= 0; i--) {
  //   if (i == height.length - 1) {
  //     dpRight[i] = height[i];
  //   } else {
  //     dpRight[i] = max(dpRight[i + 1], height[i]);
  //   }
  // }
  // for (int i = 1; i < height.length - 1; i++) {
  //   result += min(dpLeft[i], dpRight[i]) - height[i];
  // }
  // return result;
// 单调栈，栈内存储单调递减的数据，遇到比栈顶元素大的元素时，表示栈顶元素形成了一个凹槽，可以接雨水
  int res = 0;
  List<int> stack = [];
  stack.add(0);
  for (int i = 1; i < height.length; i++) {
    if (height[i] < height[stack.last]) {
      stack.add(i);
    } else if (height[i] == height[stack.last]) {
      stack.removeLast();
      stack.add(i);
    } else {
      while (stack.isNotEmpty && height[i] > height[stack.last]) {
        int top = stack.removeLast();
        if (stack.isEmpty) break;
        int height1 = min(height[stack.last], height[i]) - height[top];
        int width = i - stack.last - 1;
        res += height1 * width;
      }
      stack.add(i);
    }
  }
  return res;
}

/**
 46.全排列
力扣题目链接

给定一个 没有重复 数字的序列，返回其所有可能的全排列。

示例:

输入: [1,2,3]
输出: [ [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] ]
 */

List<List<int>> permute(List<int> nums) {
  if (nums.isEmpty) return [];
  List<List<int>> result = [];
  List<int> current = [];
  List<bool> isUsed = List.filled(nums.length, false);
  void backTrack() {
    if (current.length == nums.length) {
      result.add(List.from(current));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      if (isUsed[i]) continue;
      isUsed[i] = true;
      current.add(nums[i]);
      backTrack();
      current.removeLast(); // backtrack
      isUsed[i] = false; // mark as unused
    }
  }

  backTrack();
  return result;
}

List<List<int>> permuteUnique(List<int> nums) {
  nums.sort(); // Sort to handle duplicates
  if (nums.isEmpty) return [];
  List<List<int>> result = [];
  List<int> current = [];
  List<bool> isUsed = List.filled(nums.length, false);
  void backTrack() {
    if (current.length == nums.length) {
      result.add(List.from(current));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      if (isUsed[i]) continue;
      if (i > 0 && nums[i] == nums[i - 1] && !isUsed[i - 1]) continue;
      isUsed[i] = true;
      current.add(nums[i]);
      backTrack();
      current.removeLast(); // backtrack
      isUsed[i] = false; // mark as unused
    }
  }

  backTrack();
  return result;
}

void rotate(List<List<int>> matrix) {
  int top = 0;
  int bottom = matrix.length - 1;
  while (top < bottom) {
    for (int j = 0; j < matrix[0].length; j++) {
      int temp = matrix[top][j];
      matrix[top][j] = matrix[bottom][j];
      matrix[bottom][j] = temp;
    }
    top++;
    bottom--;
  }
  for (int i = 0; i < matrix.length; i++) {
    for (int j = i + 1; j < matrix[0].length; j++) {
      int temp = matrix[i][j];
      matrix[i][j] = matrix[j][i];
      matrix[j][i] = temp;
    }
  }
}

/**
 * 49. 字母异位词分组
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
 */
// 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
// 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
List<List<String>> groupAnagrams(List<String> strs) {
  Map<String, List<String>> map = {};
  for (int i = 0; i < strs.length; i++) {
    List<String> temp = strs[i].split('');
    temp.sort();
    String key = temp.join('');
    if (!map.containsKey(key)) {
      map[key] = [];
    }
    map[key]?.add(strs[i]);
  }

  return map.values.toList();
}

/**
 438. 找到字符串中所有字母异位词
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

 

示例 1:

输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
 示例 2:

输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。
 */
List<int> findAnagrams(String s, String p) {
  // 思路：维持sCount的大小为p的长度（左边移除，右边加入），然后比较两个count数组是否相等
  List<int> result = [];
  if (p.length > s.length) return result;
  List<int> pCount = List.filled(26, 0);
  List<int> sCount = List.filled(26, 0);
  for (int i = 0; i < p.length; i++) {
    pCount[p.codeUnits[i] - 'a'.codeUnitAt(0)]++;
    sCount[s.codeUnits[i] - 'a'.codeUnitAt(0)]++;
  }
  if (pCount.toString() == sCount.toString()) {
    result.add(0);
  }
  for (int i = p.length; i < s.length; i++) {
    sCount[s.codeUnits[i - p.length] - 'a'.codeUnitAt(0)]--;
    sCount[s.codeUnits[i] - 'a'.codeUnitAt(0)]++;
    if (pCount.toString() == sCount.toString()) {
      result.add(i - p.length + 1);
    }
  }
  return result;
}
/**
 * 1002. 查找共用字符
 给你一个字符串数组 words ，请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符），并以数组形式返回。你可以按 任意顺序 返回答案。
输入：words = ["bella","label","roller"] 输出：["e","l","l"]
输入：words = ["cool","lock","cook"] 输出：["c","o"]
1 <= words.length <= 100 1 <= words[i].length <= 100 words[i] 由小写英文字母组成
 */

List<String> commonChars(List<String> words) {
  //思路： 先统计words【0】的字符频次，然后统计words【i】的字符频次，然后取最小值（如果为0则表示该字符没有即在words[0]中出现，又在words[i]中出现），后续同理比较words剩余的字符串
  List<int> pCount = List.filled(26, 0);
  List<int> sCount = List.filled(26, 0);
  List<String> result = [];
  for (int i = 0; i < words[0].length; i++) {
    pCount[words[0].codeUnits[i] - 'a'.codeUnitAt(0)]++;
  }

  for (int i = 1; i < words.length; i++) {
    sCount = List.filled(26, 0);
    for (int j = 0; j < words[i].length; j++) {
      sCount[words[i].codeUnits[j] - 'a'.codeUnitAt(0)]++;
    }
    for (int j = 0; j < 26; j++) {
      pCount[j] = min(pCount[j], sCount[j]);
    }
  }
  for (int i = 0; i < 26; i++) {
    while (pCount[i] > 0) {
      result.add(String.fromCharCode(i + 'a'.codeUnitAt(0)));
      pCount[i]--;
    }
  }
  return result;
}
/*
349. 两个数组的交集
给定两个数组 nums1 和 nums2 ，返回 它们的 交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。
示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]
示例 2：
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]
解释：[4,9] 也是可通过的
*/

List<int> intersection(List<int> nums1, List<int> nums2) {
  // 思路：使用两个set分别存储两个数组的元素，然后遍历其中一个set，判断另一个set是否包含该元素，如果包含则加入结果集
  Set<int> set1 = {};
  Set<int> set2 = {};
  List<int> result = [];
  for (int i = 0; i < nums1.length; i++) {
    set1.add(nums1[i]);
  }
  for (int i = 0; i < nums2.length; i++) {
    set2.add(nums2[i]);
  }
  for (int num in set1) {
    if (set2.contains(num)) {
      result.add(num);
    }
  }
  return result;
}

/*
350. 两个数组的交集 II
给你两个整数数组 nums1 和 nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，
应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
示例 2:

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
1 <= nums1.length, nums2.length <= 1000
0 <= nums1[i], nums2[i] <= 1000
*/
List<int> intersect(List<int> nums1, List<int> nums2) {
  // 和1002. 查找共用字符思路一样
  List<int> result = [];
  List<int> nums1Count = List.filled(1001, 0);
  List<int> nums2Count = List.filled(1001, 0);
  for (int item in nums1) {
    nums1Count[item]++;
  }
  for (int item in nums2) {
    nums2Count[item]++;
  }
  for (int i = 0; i < 1001; i++) {
    nums2Count[i] = min(nums1Count[i], nums2Count[i]);
    while (nums2Count[i] > 0) {
      result.add(i);
      nums2Count[i]--;
    }
  }
  return result;
}

//-1 -1 -1 2 2
bool isHappy(int n) {
  //思路：n==1则是快乐数，否则计算各位数字的平方和，如果出现循环则不是快乐数
  Set<int> seen = {};
  while (n != 1 && !seen.contains(n)) {
    seen.add(n);
    int sum = 0;
    while (n > 0) {
      int digit = n % 10;
      sum += digit * digit;
      n = n ~/ 10;
    }
    n = sum;
  }
  return n == 1;
}

/**
53. 最大子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
输入: [-2,1,1,1,-4,4,-1,2,1,-5,4]
输出: 6
解释:  连续子数组  [4,-1,2,1] 的和最大，为  6。
 */
int maxSubArray(List<int> nums) {
  List<int> dp = List.filled(nums.length, 0);
  dp[0] = nums[0];
  int res = dp[0];
  for (int i = 1; i < nums.length; i++) {
    dp[i] = max(dp[i - 1] + nums[i], nums[i]);
    res = max(res, dp[i]);
  }
  return res;
  // int sum = 0;
  // int maxSum = nums[0];
  // int maxLeft = 0;
  // int maxRight = 0;
  // int left = 0;
  // for (int fast = 0; fast < nums.length; fast++) {
  //   sum += nums[fast];
  //   // maxSum = max(maxSum, sum);
  //   if (sum > maxSum) {
  //     maxSum = sum;
  //     maxLeft = left;
  //     maxRight = fast;
  //   }
  //   if (sum < 0) {
  //     left = fast + 1;
  //     sum = 0; // reset sum if it becomes negative
  //   }
  // }
  // // return [maxLeft,maxRight];
  // return maxSum;
}

/*
54. 螺旋矩阵
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
*/
List<int> spiralOrder(List<List<int>> matrix) {
  //思路是维护四个边界，left,right,top,bottom
  /*
a. 从左到右遍历上边界：从 left 到 right，访问 matrix[top][j]，然后 top++（因为这一行已经处理完）。

b. 从上到下遍历右边界：从 top 到 bottom，访问 matrix[i][right]，然后 right--（因为这一列已经处理完）。

c. 从右到左遍历下边界：但前提是 top <= bottom（因为可能没有行了），从 right 到 left，访问 matrix[bottom][j]，然后 bottom--。

d. 从下到上遍历左边界：但前提是 left <= right（因为可能没有列了），从 bottom 到 top，访问 matrix[i][left]，然后 left++。
*/
  int left = 0,
      right = matrix[0].length - 1,
      top = 0,
      bottom = matrix.length - 1;
  List<int> result = [];
  int x = left, y = top;

  while (left <= right && top <= bottom) {
    for (x = left; x <= right; x++) result.add(matrix[top][x]);
    top++;
    for (y = top; y <= bottom; y++) result.add(matrix[y][right]);
    right--;
    if (top <= bottom) {
      for (x = right; x >= left; x--) result.add(matrix[bottom][x]);
      bottom--;
    }
    if (left <= right) {
      for (y = bottom; y >= top; y--) result.add(matrix[y][left]);
      left++;
    }
  }
  return result;
}

/*
给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]
 */

List<List<int>> generateMatrix(int n) {
  int left = 0, right = n - 1, top = 0, bottom = n - 1;
  List<List<int>> result = List.generate(n, (_) => List.filled(n, 0));
  int num = 1;
  int x = left, y = top;
  while (left <= right && top <= bottom) {
    for (x = left; x <= right; x++) result[top][x] = num++;
    top++;
    for (y = top; y <= bottom; y++) result[y][right] = num++;
    right--;
    if (top <= bottom) {
      for (x = right; x >= left; x--) result[bottom][x] = num++;
      bottom--;
    }
    if (left <= right) {
      for (y = bottom; y >= top; y--) result[y][left] = num++;
      left++;
    }
  }
  return result;
}

/**
给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。

输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]

 */
ListNode? removeElements(ListNode? head, int val) {
  ListNode? dummy = ListNode(0, head);
  ListNode? prev = dummy;
  ListNode? current = head;
  while (current != null) {
    if (current.val != val) {
      prev = prev?.next;
      current = current.next;
    } else {
      prev?.next = current.next;
      current = current.next;
    }
  }
  return dummy.next;
}

// 反转链表
ListNode? reverseList(ListNode? head) {
  ListNode? prev = null;
  ListNode? current = head;
  while (current != null) {
    ListNode? nextTemp = current.next;
    current.next = prev;
    prev = current;
    current = nextTemp;
  }
  return prev;
}

/**
 45.跳跃游戏 II
给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳  1  步，然后跳  3  步到达数组的最后一个位置。
说明: 假设你总是可以到达数组的最后一个位置。
 */

int jump(List<int> nums) {
  if (nums.length <= 1) return 0;
  int result = 0;
  int currentDistance = 0;
  int nextDistance = 0;
  for (int i = 0; i < nums.length; i++) {
    nextDistance = max(nextDistance, i + nums[i]);
    if (i == currentDistance) {
      result++;
      currentDistance = nextDistance;
      if (currentDistance == nums.length - 1) return result;
    }
  }
  return result;
}

/**
 55. 跳跃游戏
给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
 */
bool canJump(List<int> nums) {
  if (nums.isEmpty) return true;
  int currentDistance = 0;
  int nextDistance = 0;
  for (int i = 0; i < nums.length; i++) {
    nextDistance = max(nextDistance, i + nums[i]);
    if (i == currentDistance) {
      currentDistance = nextDistance;
    } else if (i > currentDistance) {
      return false; // cannot reach this index
    }
  }
  return true;
}

/**
 56. 合并区间
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，
该数组需恰好覆盖输入中的所有区间 。
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
 */
List<List<int>> merge(List<List<int>> intervals) {
  if (intervals.length <= 1) return intervals;
  intervals.sort((a, b) => a[0].compareTo(b[0]));
  int slow = 0;
  int fast = 1;
  while (fast < intervals.length) {
    if (intervals[slow][1] < intervals[fast][0]) {
      slow++;
      intervals[slow] = intervals[fast];
      fast++;
    } else {
      intervals[slow] = [
        intervals[slow][0],
        max(intervals[fast][1], intervals[slow][1])
      ];
      fast++;
    }
  }
  return intervals.sublist(0, slow + 1);
}

/**
 738.单调递增的数字
力扣题目链接

给定一个非负整数 N，找出小于或等于 N 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。

（当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。）

示例 1:

输入: N = 10
输出: 9
示例 2:

输入: N = 1234
输出: 1234
 */
int monotoneIncreasingDigits(int n) {
  List<int> list = n.toString().split('').map((e) => int.parse(e)).toList();

  for (int i = list.length - 1; i > 0; i--) {
    if (list[i] < list[i - 1]) {
      list[i - 1]--;
      for (int j = i; j < list.length; j++) {
        list[j] = 9;
      }
    }
  }
  int res = int.parse(list.join(''));
  return res;
}

/*
1005.K次取反后最大化的数组和
给定一个整数数组 A，我们只能用以下方法修改该数组：我们选择某个索引 i 并将 A[i] 替换为 -A[i]，然后总共重复这个过程 K 次。（我们可以多次选择同一个索引 i。）

以这种方式修改数组后，返回数组可能的最大和。

示例 1：

输入：A = [4,2,3], K = 1
输出：5
解释：选择索引 (1) ，然后 A 变为 [4,-2,3]。

输入：A = [2,-3,-1,5,-4], K = 2
-1 2 -3 4 5
输出：13
*/
int largestSumAfterKNegations(List<int> nums, int k) {
  nums.sort((a, b) => a.abs().compareTo(b.abs()));
  for (int i = nums.length - 1; i >= 0; i--) {
    if (nums[i] < 0 && k > 0) {
      nums[i] = -nums[i];
      k--;
    }
  }
  while (k > 0) {
    nums[0] = -nums[0];
    k--;
  }
  int res = nums.reduce((a, b) => a + b);
  return res;
}

List<List<int>> insert(List<List<int>> intervals, List<int> newInterval) {
  intervals.add(newInterval);
  intervals.sort((a, b) => a[0].compareTo(b[0]));
  if (intervals.length + 1 <= 1) return intervals;
  int slow = 0;
  int fast = 1;
  while (fast < intervals.length) {
    if (intervals[slow][1] < intervals[fast][0]) {
      slow++;
      intervals[slow] = intervals[fast];
      fast++;
    } else {
      intervals[slow] = [
        intervals[slow][0],
        max(intervals[slow][1], intervals[fast][1])
      ];
      fast++;
    }
  }
  return intervals.sublist(0, slow + 1);
}

int lengthOfLastWord(String s) {
  int slow = s.length - 1;
  int fast = s.length - 1;
  while (fast >= 0) {
    if (s[fast] == ' ' && fast != slow) break;
    if (s[fast] == ' ') {
      slow--;
      fast--;
    } else {
      fast--;
    }
  }
  return slow - fast;
}

ListNode? rotateRight(ListNode? head, int k) {
  int nodeCount = 0;
  ListNode? current = head;
  while (current != null) {
    nodeCount++;
    current = current.next;
  }
  current = head;
  k = k % nodeCount; // handle cases where k is larger than nodeCount
  if (head == null || head.next == null || k == 0) return head;
  ListNode? newTail = head;
  ListNode? newHead = head;
  for (int i = 0; i < k; i++) {
    current = current?.next;
  }
  while (current?.next != null) {
    newTail = newTail?.next;
    current = current?.next;
  }
  newHead = newTail?.next; // new head is the next of new tail
  newTail?.next = null; // break the link to make new tail the end of the list
  current?.next = head; // link the old head to the end of the new list
  return newHead;
}

/**
 62. 不同路径
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

输入：m = 3, n = 7
输出：28
 */
int uniquePaths(int m, int n) {
  if (m <= 0 || n <= 0) return 0; // handle edge cases
  List<List<int>> dp = List.filled(m, List.filled(n, 0));
  for (int i = 0; i < m; i++) {
    dp[i][0] = 1; // only one way to reach the first column
  }
  for (int j = 0; j < n; j++) {
    dp[0][j] = 1; // only one way to reach the first row
  }
  for (int i = 1; i < m; i++) {
    for (int j = 1; j < n; j++) {
      dp[i][j] = dp[i - 1][j] + dp[i][j - 1]; // sum of ways from top and left
    }
  }
  return dp[m - 1][n - 1]; // return the bottom-right corner value
}

/**
 63. 不同路径 II
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 1 和 0 来表示。

输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
 */
int uniquePathsWithObstacles(List<List<int>> obstacleGrid) {
  List<List<int>> dp = List.generate(
      obstacleGrid.length, (_) => List.filled(obstacleGrid[0].length, 0));
  if (obstacleGrid.length <= 0 || obstacleGrid.length <= 0)
    return 0; // handle edge cases

  for (int i = 0; i < obstacleGrid.length; i++) {
    if (obstacleGrid[i][0] == 1) break;
    dp[i][0] = 1;
  }
  for (int i = 0; i < obstacleGrid[0].length; i++) {
    if (obstacleGrid[0][i] == 1) break;
    dp[0][i] = 1;
  }
  for (int i = 1; i < obstacleGrid.length; i++) {
    for (int j = 1; j < obstacleGrid[0].length; j++) {
      if (obstacleGrid[i][j] == 1) {
        dp[i][j] = 0; // obstacle
      } else {
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1]; // sum of ways from top and left
      }
    }
  }
  return dp[obstacleGrid.length - 1][obstacleGrid[0].length - 1];
}

int minPathSum(List<List<int>> grid) {
  int m = grid.length;
  int n = grid[0].length;
  if (m <= 0 || n <= 0) return 0; // handle edge cases
  List<List<int>> dp = List.generate(m, (_) => List.filled(n, 0));
  dp[0][0] = grid[0][0]; // start point
  for (int i = 1; i < n; i++) {
    dp[0][i] = dp[0][i - 1] + grid[0][i]; // fill first row
  }
  for (int i = 1; i < m; i++) {
    dp[i][0] = dp[i - 1][0] + grid[i][0]; // fill first column
  }
  for (int i = 1; i < m; i++) {
    for (int j = 1; j < n; j++) {
      dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]; // fill the rest
    }
  }
  return dp[m - 1][n - 1]; // return the bottom-right corner value
}

List<int> plusOne(List<int> digits) {
  if (digits.isEmpty) return [1];
  int len = digits.length - 1;
  int carry = 1; // start with carry of 1 for the plus one operation
  do {
    int sum = digits[len] + carry;
    carry = sum ~/ 10; // calculate new carry
    digits[len] = sum % 10; // update the digit
    len--;
  } while (carry > 0 && len >= 0);
  if (carry > 0) {
    digits.insert(0, carry); // if there's still carry, insert it at the front
  }
  return digits;
}

String addBinary(String a, String b) {
  if (a.isEmpty) return b;
  if (b.isEmpty) return a;
  int indexA = a.length - 1;
  int indexB = b.length - 1;
  int upTen = 0;
  String result = '';
  while (indexA >= 0 || indexB >= 0 || upTen > 0) {
    int numA = indexA >= 0
        ? a[indexA] == '1'
            ? 1
            : 0
        : 0;
    int numB = indexB >= 0
        ? b[indexB] == '1'
            ? 1
            : 0
        : 0;
    int sum = numA + numB + upTen;
    upTen = sum ~/ 2; // calculate carry
    result = (sum % 2 == 1 ? '1' : '0') + result; // append the result
    indexA--;
    indexB--;
  }
  return result;
}

int mySqrt(int x) {
  int left = 0;
  int right = x;
  while (left <= right) {
    int mid = (left + right) ~/ 2;
    if (mid * mid == x) {
      return mid; // found exact square root
    } else if (mid * mid < x) {
      left = mid + 1; // search in the right half
    } else {
      right = mid - 1; // search in the left half
    }
  }
  return right; // right will be the largest integer whose square is less than or equal to x
}

/**
 70. 爬楼梯
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶

 */
int climbStairs(int n) {
  if (n <= 1) return n; // only one way to climb one step
  List<int> dp = List.filled(n, 0);
  dp[0] = 1;
  dp[1] = 2;
  for (int i = 2; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2];
  }
  return dp[n - 1]; // return the number of ways to reach the nth step
}

/**
 746. 使用最小花费爬楼梯
给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。

请你计算并返回达到楼梯顶部的最低花费。
输入：cost = [10,15,20]
输出：15
 */
int minCostClimbingStairs(List<int> cost) {
  List<int> dp = List.filled(cost.length + 1, 0);
  dp[0] = 0;
  dp[1] = 0;
  for (int i = 2; i <= cost.length; i++) {
    dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
  }
  return dp[cost.length];
}

String simplifyPath(String path) {
  List<String> stack = path.split('/');
  List<String> result = [];
  for (String dir in stack) {
    if (dir == '' || dir == '.') continue; // skip empty or current directory
    if (dir == '..') {
      if (result.isNotEmpty) result.removeLast(); // go up one directory
    } else {
      result.add(dir); // add valid directory to the stack
    }
  }
  return '/' + result.join('/'); // join the result with '/' and return
}

void setZeroes(List<List<int>> matrix) {
  bool rowZero = false;
  bool colZero = false;
  for (int i = 0; i < matrix.length; i++) {
    if (matrix[i][0] == 0) {
      colZero = true; // first column has zero
      break;
    }
  }
  for (int j = 0; j < matrix[0].length; j++) {
    if (matrix[0][j] == 0) {
      rowZero = true; // first row has zero
      break;
    }
  }
  for (int i = 1; i < matrix.length; i++) {
    for (int j = 1; j < matrix[0].length; j++) {
      if (matrix[i][j] == 0) {
        matrix[i][0] = 0; // mark first column
        matrix[0][j] = 0; // mark first row
      }
    }
  }
  for (int i = 1; i < matrix.length; i++) {
    for (int j = 1; j < matrix[0].length; j++) {
      if (matrix[i][0] == 0 || matrix[0][j] == 0) {
        matrix[i][j] = 0; // set to zero if marked
      }
    }
  }
  if (rowZero) {
    for (int j = 0; j < matrix[0].length; j++) {
      matrix[0][j] = 0; // set first row to zero
    }
  }
  if (colZero) {
    for (int i = 0; i < matrix.length; i++) {
      matrix[i][0] = 0; // set first column to zero
    }
  }
}

bool searchMatrix(List<List<int>> matrix, int target) {
  if (matrix.isEmpty || matrix[0].isEmpty) return false;
  int left = 0;
  int right = matrix.length * matrix[0].length - 1;
  while (left <= right) {
    int mid = (left + right) ~/ 2;
    int i = mid ~/ matrix[0].length; // row index
    int j = mid % matrix[0].length; // column index
    if (matrix[i][j] == target) {
      return true; // found the target
    } else if (matrix[i][j] < target) {
      left = mid + 1; // search in the right half
    } else {
      right = mid - 1; // search in the left half
    }
  }
  return false; // target not found
}

void sortColors(List<int> nums) {
  if (nums.length <= 1) return; // no need to sort if length is 0 or 1
  int zeroIndex = 0;
  int oneIndex = 0;
  for (int i = 0; i < nums.length; i++) {
    if (nums[i] == 0) {
      int temp = nums[zeroIndex];
      nums[zeroIndex] = nums[i];
      nums[i] = temp;
      zeroIndex++;
    }
  }
  oneIndex = zeroIndex;
  for (int i = zeroIndex; i < nums.length; i++) {
    if (nums[i] == 1) {
      int temp = nums[oneIndex];
      nums[oneIndex] = nums[i];
      nums[i] = temp;
      oneIndex++;
    }
  }
}

String minWindow(String s, String t) {
  int left = 0;
  int right = 0;
  String result = '';
  int minLen = s.length + 1;
  Map<String, int> tCount = {};
  for (int i = 0; i < t.length; i++) {
    String char = t[i];
    if (!tCount.containsKey(char)) {
      tCount[char] = 0;
    }
    tCount[char] = tCount[char]! + 1; // count characters in t
  }
  while (right < s.length) {
    String char = s[right];
    if (tCount.containsKey(char)) {
      tCount[char] = tCount[char]! - 1;
    }
    right++;
    while (tCount.values.every((count) => count <= 0)) {
      // Check if all characters in t are present in the current window
      if (minLen > right - left) {
        minLen = right - left;
        result = s.substring(left, right);
      }
      String leftChar = s[left];
      if (tCount.containsKey(leftChar)) {
        tCount[leftChar] = tCount[leftChar]! + 1;
      }
      left++;
    }
  }
  return result;
}

/**
 77. 组合
给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。

你可以按 任何顺序 返回答案。
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
 */
List<List<int>> combine(int n, int k) {
  List<List<int>> result = [];
  List<int> current = [];
  void backTrack(int start) {
    if (current.length == k) {
      result.add(List.from(current));
      return;
    }
    if (current.length + (n - start + 1) < k)
      return; // not enough elements left to fill k
    for (int i = start; i <= n; i++) {
      current.add(i);
      backTrack(i + 1); // move to the next number
      current.removeLast(); // backtrack
    }
  }

  backTrack(1);
  return result;
}

/**
 216.组合总和III
找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
所有数字都是正整数。
解集不能包含重复的组合。
示例 1: 输入: k = 3, n = 7 输出: [[1,2,4]]
示例 2: 输入: k = 3, n = 9 输出: [[1,2,6], [1,3,5], [2,3,4]]
 */
// k=9, n=45
List<List<int>> combinationSum3(int k, int n) {
  List<List<int>> res = [];
  List<int> path = [];
  int sum = 0;
  void backTrack(int start) {
    if (sum > n) return; // 剪枝
    if (path.length == k && sum == n) {
      res.add(List.from(path));
      return;
    }
    if (9 - start + 1 < k - path.length) return; // 剪枝
    for (int i = start; i <= 9; i++) {
      path.add(i);
      sum += i;
      backTrack(i + 1); // move to the next number
      path.removeLast(); // backtrack
      sum -= i;
    }
  }

  backTrack(1);
  return res;
}

/**
 78.子集
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：解集不能包含重复的子集。

示例: 输入: nums = [1,2,3] 输出: [ [3],   [1],   [2],   [1,2,3],   [1,3],   [2,3],   [1,2],   [] ]
 */
List<List<int>> subsets(List<int> nums) {
  List<List<int>> result = [];
  List<int> current = [];
  void backTrack(int start) {
    result.add(List.from(current));

    for (int i = start; i < nums.length; i++) {
      current.add(nums[i]);
      backTrack(i + 1); // move to the next number
      current.removeLast(); // backtrack
    }
  }

  backTrack(0);
  return result;
}

// 删除元素，最多保留两个相同的元素
int removeDuplicates2(List<int> nums) {
  if (nums.length <= 2) return nums.length; // no duplicates to remove
  int slow = 0, fast = 2;
  for (int fast = slow; fast < nums.length; fast++) {
    if (nums[fast] != nums[slow]) {
      nums[slow] = nums[fast];
      slow++;
    }
  }
  return slow;
  // [1,1,1,1,1,1,2,3,3]
}

bool search2(List<int> nums, int target) {
  nums.sort();
  int left = 0, right = nums.length - 1;
  while (left <= right) {
    int mid = (left + right) ~/ 2;
    if (nums[mid] == target)
      return true;
    else if (nums[mid] < target)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return false;
}

ListNode? deleteDuplicates(ListNode? head) {
  if (head == null || head.next == null) return head; // no duplicates
  ListNode? dummy = ListNode(0, head);
  ListNode? pre = dummy, cur = head, last = head.next;
  bool needDelete = false;
  while (last != null) {
    if (cur?.val == last.val) {
      needDelete = true;
      last = last.next;
    } else {
      if (needDelete) {
        pre!.next = last;
        cur = last;
        last = last.next;
      } else {
        last = last.next;
        cur = cur?.next;
        pre = pre?.next;
      }
      needDelete = false; // reset flag
    }
  }
  if (needDelete) {
    pre!.next = last;
    cur = last;
  }
  return dummy.next;
}

ListNode? deleteDuplicates2(ListNode? head) {
  if (head == null || head.next == null) return head; // no duplicates
  ListNode? dummy = ListNode(0, head);
  ListNode? cur = head, last = head.next;
  while (last != null) {
    if (cur?.val == last.val) {
      last = last.next; // skip duplicates
      if (last == null) {
        cur!.next = null; // if last is null, set cur's next to null
      }
    } else {
      cur!.next = last;
      cur = cur.next;
      last = last.next; // move to next node
    }
  }
  return dummy.next; // return the new head
}

/**
 84. 柱状图中最大的矩形
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
 */
int largestRectangleArea(List<int> heights) {
  // 使用单调栈，栈内存储单调递增的数据，也就是栈头是最大值
  // 然后遍历数组，如果当前元素小于栈顶元素，就表示栈顶元素所能组成的最大高度的矩形结束了
  if (heights.length == 0) return 0;
  List<int> stack = [];
  int result = 0;
  stack.add(-1); // add a sentinel value to handle edge cases
  for (int i = 0; i < heights.length; i++) {
    while (stack.last != -1 && heights[i] <= heights[stack.last]) {
      //当前元素小于栈顶元素
      int height = heights[stack.removeLast()];
      int width = i - stack.last - 1; //i和stack.peek()为stack.pop()元素的左右两个小于元素
      result = max(result, height * width);
    }
    stack.add(i);
  }
  // 栈顶元素所能组成的最大高度的矩形也已经结束
  while (stack.last != -1) {
    int height = heights[stack.removeLast()];
    int width = heights.length - stack.last - 1; // 处理剩余元素
    result = max(result, height * width);
  }
  return result;
}

/**
 86. 分隔链表
给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。
 */
ListNode? partition(ListNode? head, int x) {
  ListNode? lessHead = ListNode(0);
  ListNode? greaterHead = ListNode(0);
  ListNode dummyLess = lessHead;
  ListNode dummyGreater = greaterHead;
  while (head != null) {
    if (head.val < x) {
      lessHead!.next = head;
      lessHead = lessHead.next;
    } else {
      greaterHead!.next = head;
      greaterHead = greaterHead.next;
    }
    head = head.next; // move to the next node
  }
  lessHead!.next = dummyGreater
      .next; // link less than x list with greater than or equal to x list
  greaterHead?.next = null;
  return dummyLess.next; // return the new head of the partitioned list
}

/**
 131.分割回文串
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。

示例: 输入: "aab" 输出: [ ["aa","b"], ["a","a","b"] ]
 */
List<List<String>> partition2(String s) {
  List<List<String>> result = [];
  List<String> current = [];
  bool isPalindrome(String str) {
    int left = 0, right = str.length - 1;
    while (left < right) {
      if (str[left] != str[right]) return false; // characters do not match
      left++;
      right--;
    }
    return true; // all characters match
  }

  void backTrack(int start) {
    if (start == s.length) {
      result.add(List.from(current)); // add the current partition to result
      return;
    }
    for (int i = start; i < s.length; i++) {
      String substring = s.substring(start, i + 1);
      if (isPalindrome(substring)) {
        current.add(substring); // add valid palindrome substring
        backTrack(i + 1); // move to the next index
        current.removeLast(); // backtrack
      }
    }
  }

  backTrack(0); // start backtracking from index 0
  return result;
}

void merge2(List<int> nums1, int m, List<int> nums2, int n) {
  for (int i = nums1.length - 1; i >= 0; i--) {
    if (n - 1 < 0 || m - 1 >= 0 && nums1[m - 1] > nums2[n - 1]) {
      nums1[i] = nums1[m - 1];
      m--;
    } else {
      nums1[i] = nums2[n - 1];
      n--;
    }
  }
}

List<int> grayCode(int n) {
  List<int> ret = [];
  for (int i = 0; i < 1 << n; i++) {
    ret.add((i >> 1) ^ i);
  }
  return ret;
}

/**
 90.子集II
给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：解集不能包含重复的子集。

示例:

输入: [1,2,2]
输出: [ [2], [1], [1,2,2], [2,2], [1,2], [] ]
 */
//  输入：nums = [1,2,2]
// 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
List<List<int>> subsetsWithDup(List<int> nums) {
  List<List<int>> result = [];
  List<int> current = [];
  // List<bool> isUsed = List.filled(nums.length, false);

  nums.sort(); // Sort to handle duplicates
  void backTrack(int index) {
    result.add(List.from(current));

    for (int i = index; i < nums.length; i++) {
      if (i > index && nums[i] == nums[i - 1]) continue;
      current.add(nums[i]);
      // isUsed[i] = true; // mark as used
      backTrack(i + 1); // move to the next number
      current.removeLast(); // backtrack
      // isUsed[i] = false; // mark as unused
    }
  }

  backTrack(0);
  return result;
}

/**
 491.递增子序列
给定一个整型数组, 你的任务是找到所有该数组的递增子序列，递增子序列的长度至少是2。


4, 7, 6, 7
示例:

输入: [4, 6, 7, 7]
输出: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]
说明:

给定数组的长度不会超过15。
数组中的整数范围是 [-100,100]。
给定数组中可能包含重复数字，相等的数字应该被视为递增的一种情况。
 */
// [[4,6,7,7],[4,7,7],[6,7,7],[7,7]]
// [[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]
List<List<int>> findSubsequences(List<int> nums) {
  List<List<int>> result = [];
  List<int> current = [];
  void backTrack(int start) {
    if (current.length >= 2) {
      result.add(List.from(current));
    }
    if (start == nums.length) {
      return;
    }
    Set<int> usedInLevel = {};
    for (int i = start; i < nums.length; i++) {
      if (usedInLevel.contains(nums[i]))
        continue; // skip duplicates in the same recursion level
      if (current.isNotEmpty && nums[i] < current.last)
        continue; // ensure non-decreasing order
      usedInLevel.add(nums[i]);
      current.add(nums[i]);
      backTrack(i + 1);
      current.removeLast();
    }
  }

  backTrack(0);
  return result;
}

// "2101"
// "1123"
// 1 1 / 1 11
// 1 1 2 / 11 2 / 1 12
// 1 1 2 3 / 11 2 3 / 1 12 3 / 1 1 23 / 11 23
int numDecodings(String s) {
  int n = s.length;
  List<int> f = List.filled(n + 1, 0);
  f[0] = 1;
  for (int i = 1; i <= n; ++i) {
    if (s[i - 1] != '0') {
      f[i] += f[i - 1];
    }
    if (i > 1 &&
        s[i - 2] != '0' &&
        ((s[i - 2] == '1' || s[i - 2] == '2') &&
            s[i - 1].codeUnitAt(0) <= '6'.codeUnitAt(0))) {
      f[i] += f[i - 2];
    }
  }
  return f[n];
}

// 1 2 3 4 5
// 1 4 3 2 5
ListNode? reverseBetween(ListNode? head, int left, int right) {
  if (head == null || head.next == null)
    return head; // no need to reverse if list is empty or has one node
  ListNode? dummy = ListNode(0, head);
  ListNode? pre = dummy, cur = head, last = head;
  for (int i = 0; i < left - 1; i++) {
    pre = pre?.next; // move pre to the node before left
  }
  cur = pre?.next; // cur is the left node
  last = cur?.next; // last is the node after left
  for (int i = 0; i < right - left; i++) {
    ListNode? next = last?.next; // save the next node
    last?.next = cur; // reverse the link
    cur = last; // move cur to the last node
    last = next; // move last to the next node
  }
  pre?.next?.next = last;
  pre?.next = cur; // link the pre node to the new head
  return dummy.next; // return the new head of the reversed list
}

/**
 93.复原IP地址
给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
有效的 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
例如："0.1.2.201" 和 "192.168.1.1" 是 有效的 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效的 IP 地址。

示例 1：

输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]
示例 2：

输入：s = "0000"
输出：["0.0.0.0"]
示例 3：

输入：s = "1111"
输出：["1.1.1.1"]
示例 4：

输入：s = "010010"
输出：["0.10.0.10","0.100.1.0"]
示例 5：

输入：s = "101023"
输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
提示：

0 <= s.length <= 3000
s 仅由数字组成
 */
// 255.255.11.135
// "25525511135"
List<String> restoreIpAddresses(String s) {
  // List<String> result = [];
  // List<String> current = [];
  // void backTrack(int start) {
  //   if (start == s.length && current.length == 4) {
  //     result.add(current.join('.'));
  //     return;
  //   }
  //   if (current.length >= 4 || start >= s.length) return; // invalid state
  //   for (int i = start; i < s.length; i++) {
  //     if ((s.length - i) ~/ 3 + current.length > 4 ||
  //         s.length - i + current.length < 4) return; // not enough segments left
  //     String segment = s.substring(start, i + 1);
  //     if (segment.length > 3 ||
  //         (segment.length > 1 && segment[0] == '0') ||
  //         int.parse(segment) > 255) {
  //       break; // invalid segment
  //     }
  //     current.add(segment); // add valid segment
  //     backTrack(i + 1); // move to the next segment
  //     current.removeLast(); // backtrack
  //   }
  // }

  // backTrack(0);
  // return result;
  bool isValid(String segment) {
    if (segment.length > 3 ||
        (segment.length > 1 && segment[0] == '0') ||
        int.parse(segment) > 255) {
      return false; // invalid segment
    }
    return true;
  }

  List<String> result = [];
  List<String> current = [];
  void backTrack(int start) {
    if (current.length == 4 && start == s.length) {
      result.add(current.join('.'));
      return;
    }
    if (current.length > 4 || start >= s.length) return; // invalid state
    for (int i = start; i < s.length; i++) {
      String segment = s.substring(start, i + 1);
      if (isValid(segment)) {
        current.add(segment); // add valid segment
        backTrack(i + 1); // move to the next segment
        current.removeLast(); // backtrack
      }
    }
  }

  backTrack(0);
  return result;
}

List<int> preOrder(TreeNode? root) {
  List<int> result = [];
  void backTrack(TreeNode? node) {
    if (node == null) return;
    result.add(node.val); // visit the root
    backTrack(node.left); // traverse left subtree
    backTrack(node.right); // traverse right subtree
  }

  backTrack(root);
  return result;
}

List<int> inOrder(TreeNode? root) {
  List<int> reset = [];
  void backTrack(TreeNode? node) {
    if (node == null) return;
    backTrack(node.left); // traverse left subtree
    reset.add(node.val); // visit the root
    backTrack(node.right); // traverse right subtree
  }

  backTrack(root);
  return reset;
}

List<int> inOrder2(TreeNode? root) {
  List<int> result = [];
  List<TreeNode> stack = [];
  TreeNode? cur = root;
  while (cur != null || stack.length > 0) {
    if (cur != null) {
      stack.add(cur); // traverse left subtree
      cur = cur.left; // move to the left child
    } else {
      cur = stack.removeLast();
      result.add(cur.val); // visit the root
      cur = cur.right; // move to the right child
    }
  }
  return result;
}

List<int> postOrder(TreeNode? root) {
  List<int> result = [];
  void backTrack(TreeNode? node) {
    if (node == null) return;
    backTrack(node.left); // traverse left subtree
    backTrack(node.right); // traverse right subtree
    result.add(node.val); // visit the root
  }

  backTrack(root);
  return result;
}

// 123 - 2 3 1
// 1 3 2
List<int> preOrder2(TreeNode? root) {
  if (root == null) return [];
  List<int> result = [];
  List<TreeNode> stack = [];
  stack.add(root);
  while (stack.length > 0) {
    TreeNode? node = stack.removeLast();
    result.add(node.val); // visit the root
    if (node.right != null) {
      stack.add(node.right!);
    }
    if (node.left != null) {
      stack.add(node.left!);
    }
  }
  return result;
}

List<int> postOrder2(TreeNode? root) {
  if (root == null) return [];
  List<int> result = [];
  List<TreeNode> stackIn = [];
  List<TreeNode> stackOut = [];
  stackIn.add(root);
  while (stackIn.length > 0) {
    TreeNode node = stackIn.removeLast();
    stackOut.add(node); // add to output stack
    if (node.left != null) {
      stackIn.add(node.left!); // add left child to input stack
    }
    if (node.right != null) {
      stackIn.add(node.right!); // add right child to input stack
    }
  }
  while (stackOut.length > 0) {
    TreeNode node = stackOut.removeLast();
    result.add(node.val); // visit the root
  }
  return result;
}

List<int> postOrder3(TreeNode? root) {
  if (root == null) return [];
  List<int> result = [];
  List<TreeNode> stack = [];
  stack.add(root);
  while (stack.length > 0) {
    TreeNode node = stack.removeLast();
    result.add(node.val); // visit the root
    if (node.left != null) {
      stack.add(node.left!); // add left child to stack
    }
    if (node.right != null) {
      stack.add(node.right!); // add right child to stack
    }
  }

  return result.reversed.toList(); // reverse the result to get post-order
}
//1 2 4 5 3 6 7
//4 5 2 6 7 3 1

// [3,9,20,null,null,15,7]
// [[3],[9,20],[15,7]]
List<List<int>> levelOrder(TreeNode? root) {
  if (root == null) return [];
  List<List<int>> result = [];
  List<TreeNode> queue = [];
  List<int> currentLevel = [];
  List<TreeNode> nextQueue = [];
  queue.insert(0, root);
  while (queue.length > 0 || nextQueue.length > 0) {
    if (queue.length == 0) {
      queue = nextQueue;
      result.add(currentLevel);
      currentLevel = [];
      nextQueue = [];
    }
    TreeNode? node = queue.removeLast();
    currentLevel.add(node.val); // visit the root
    if (node.left != null) {
      nextQueue.insert(0, node.left!); // add left child to next queue
    }
    if (node.right != null) {
      nextQueue.insert(0, node.right!); // add right child to next queue
    }
  }
  if (currentLevel.isNotEmpty) {
    result.add(currentLevel); // add the last level if not empty
  }
  return result;
}

// 2 3
List<List<int>> zigzagLevelOrder(TreeNode? root) {
  if (root == null) return []; // empty tree
  List<List<int>> result = [];
  List<TreeNode> queue = [];
  List<TreeNode> nextQueue = [];
  List<int> currentLevel = [];
  bool leftToRight = true; // flag to determine the direction of traversal
  queue.insert(0, root);
  while (queue.length > 0 || nextQueue.length > 0) {
    if (queue.length == 0) {
      queue = nextQueue;
      nextQueue = [];
      if (!leftToRight)
        currentLevel = currentLevel.reversed
            .toList(); // reverse the order if not left to right
      result.add(currentLevel);
      currentLevel = [];
      leftToRight = !leftToRight; // toggle the direction
    }
    TreeNode? node = queue.removeLast();
    currentLevel.add(node.val); // visit the root

    if (node.left != null) {
      nextQueue.insert(0, node.left!); // add left child to next queue
    }
    if (node.right != null) {
      nextQueue.insert(0, node.right!); // add right child to next queue
    }
  }
  if (currentLevel.isNotEmpty) {
    if (!leftToRight) {
      currentLevel = currentLevel.reversed
          .toList(); // reverse the order if not left to right
    }
    result.add(currentLevel); // add the last level if not empty
  }
  return result;
}

/**
 96.不同的二叉搜索树
给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
 */
int numTrees(int n) {
  List<int> dp = List.filled(n + 1, 0);
  if (n < 2)
    return 1; // base case: only one way to form a tree with 0 or 1 node
  dp[0] = 1; // base case: one way to form an empty tree
  dp[1] = 1; // base case: one way to form a tree with one node
  dp[2] = 2;
  for (int i = 3; i <= n; i++) {
    for (int j = 0; j < i; j++) {
      dp[i] += dp[j] * dp[i - j - 1]; // Catalan number formula
    }
  }
  return dp[n];
}

bool isValidBST(TreeNode? root) {
  // if (root == null) return true; // empty tree is a valid BST
  // List<TreeNode> stack = [];
  // int? lastValue = null; // minimum value for int
  // TreeNode? cur = root;
  // while (cur != null || stack.isNotEmpty) {
  //   if (cur != null) {
  //     stack.add(cur); // traverse left subtree
  //     cur = cur.left;
  //   } else {
  //     cur = stack.removeLast(); // visit the root
  //     if (lastValue != null && cur.val <= lastValue)
  //       return false; // check BST property
  //     lastValue = cur.val; // update last value
  //     cur = cur.right; // traverse right subtree
  //   }
  // }
  // return true; // all nodes are in valid order

  // if (root == null) return true;

  // int? max = null;
  // bool left = isValidBST(root.left);
  // if (max != null && root.val <= max) return false;
  // max = root.val;
  // bool right = isValidBST(root.right);
  // return left && right;

//[5,1,4,null,null,3,6]
  if (root == null) return true;
  int? prev;
  bool backTrack(TreeNode? node) {
    if (node == null) return true;
    bool left = backTrack(node.left);
    if (prev != null && node.val <= prev!) return false;
    prev = node.val; // update prev to current node's value
    bool right = backTrack(node.right);
    return left && right; // both left and right subtrees must be valid
  }

  return backTrack(root);
}

// 1 6 3 4 5 2 7
void recoverTree(TreeNode? root) {
  List<TreeNode> stack = [];
  List<TreeNode> swappedNodes = [];
  TreeNode? lastNode = null; // minimum value for int
  TreeNode? cur = root;
  while (cur != null || stack.isNotEmpty) {
    if (cur != null) {
      stack.add(cur); // traverse left subtree
      cur = cur.left;
    } else {
      cur = stack.removeLast();
      if (lastNode != null && cur.val <= lastNode.val) {
        swappedNodes.add(lastNode); // first swapped node
        swappedNodes.add(cur); // second swapped node
      }
      lastNode = cur; // update last node
      cur = cur.right; // traverse right subtree
    }
  }
  int temp = swappedNodes[0].val;
  swappedNodes[0].val = swappedNodes[swappedNodes.length - 1].val;
  swappedNodes[swappedNodes.length - 1].val = temp; // swap the values
}
/**
 572. 另一棵树的子树
给你两棵二叉树 root 和 subRoot 。检验 root 中是否包含和 subRoot 具有相同结构和节点值的子树。如果存在，返回 true ；否则，返回 false 。

二叉树 tree 的一棵子树包括 tree 的某个节点和这个节点的所有后代节点。tree 也可以看做它自身的一棵子树。
 */

bool isSubtree(TreeNode? root, TreeNode? subRoot) {
  if (root == null && subRoot == null) return true;
  if (root == null) return false;

  bool res = isSameTree(root, subRoot);
  if (res) return true;
  return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
}

bool isSameTree(TreeNode? p, TreeNode? q) {
  if (p == null && q == null) return true;
  if (p == null || q == null) return false; // one is null, the other is not
  if (p.val != q.val) return false; // values are different
  return isSameTree(p.left, q.left) &&
      isSameTree(p.right, q.right); // check left and right subtrees
}

bool isSymmetric(TreeNode? root) {
  bool isSameValue(TreeNode? left, TreeNode? right) {
    if (left == null && right == null) return true; // both are null
    if (left == null || right == null)
      return false; // one is null, the other is not
    if (left.val != right.val) return false; // values are different
    return isSameValue(left.left, right.right) &&
        isSameValue(left.right, right.left); // check mirrored subtrees
  }

  if (root == null) return true; // empty tree is symmetric
  return isSameValue(
      root.left, root.right); // check if left and right subtrees are symmetric
}

int maxDepth(TreeNode? root) {
  if (root == null) return 0; // empty tree has depth 0
  int leftDepth = maxDepth(root.left); // depth of left subtree
  int rightDepth = maxDepth(root.right); // depth of right subtree
  return 1 +
      max(leftDepth,
          rightDepth); // return max depth of both subtrees + 1 for the root
}

bool isBalanced(TreeNode? root) {
  int height(TreeNode? root) {
    if (root == null) return 0; // empty tree has height 0
    int leftHeight = height(root.left); // height of left subtree
    int rightHeight = height(root.right); // height of right subtree
    if (rightHeight == -1 || leftHeight == -1)
      return -1; // subtree is unbalanced
    if ((leftHeight - rightHeight).abs() > 1) return -1; // unbalanced condition
    return 1 + max(leftHeight, rightHeight); // return height of the subtree
  }

  if (root == null) return true; // empty tree is balanced
  return height(root) != -1; // check if the tree is balanced

//不能直接计算最大深度和最小深度的差值，因为平衡二叉树的定义是每个节点的左右子树高度差不超过1，而不是整个树的最大深度和最小深度之差不超过1
// [1,2,3,4,5,6,null,8]
  // int maxDepth(TreeNode? root) {
  //   if (root == null) return 0; // empty tree has depth 0
  //   int leftDepth = maxDepth(root.left); // depth of left subtree
  //   int rightDepth = maxDepth(root.right); // depth of right subtree
  //   return 1 +
  //       max(leftDepth,
  //           rightDepth); // return max depth of both subtrees + 1 for the root
  // }

  // int minDepth(TreeNode? root) {
  //   if (root == null) return 0; // empty tree has depth 0
  //   int leftDepth = minDepth(root.left); // depth of left subtree
  //   int rightDepth = minDepth(root.right); // depth of right subtree

  //   return 1 +
  //       min(leftDepth,
  //           rightDepth); // return min depth of both subtrees + 1 for the root
  // }

  // int maxDep = maxDepth(root);
  // int minDep = minDepth(root);
  // return (maxDep - minDep) <= 1; // check if the tree is
}

int minDepth(TreeNode? root) {
  if (root == null) return 0; // empty tree has depth 0
  int leftDepth = minDepth(root.left); // depth of left subtree
  int rightDepth = minDepth(root.right); // depth of right subtree
  if (leftDepth == 0 && rightDepth == 0) {
    return 1; // if one subtree is empty, return the depth of the other subtree
  }
  if (leftDepth == 0) return rightDepth + 1; // if left subtree is empty
  if (rightDepth == 0) return leftDepth + 1; // if right subtree is empty
  return 1 +
      min(leftDepth,
          rightDepth); // return min depth of both subtrees + 1 for the root
}

//二叉树：404.左叶子之和
int sumOfLeftLeaves(TreeNode? root) {
  int res = 0;
  void backTrack(TreeNode? node) {
    if (node == null) return;
    if (node.left != null &&
        node.left!.left == null &&
        node.left!.right == null) {
      res += node.left!.val;
    }
    backTrack(node.left);
    backTrack(node.right);
  }

  backTrack(root);
  return res;
}
/**
 257. 二叉树的所有路径
给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。

叶子节点 是指没有子节点的节点。
输入：root = [1,2,3,null,5]
输出：["1->2->5","1->3"]
 */

List<String> binaryTreePaths(TreeNode? root) {
  if (root == null) return [];
  List<String> res = [];
  List<int> path = [];
  void backTrack(TreeNode node) {
    path.add(node.val);
    if (node.left == null && node.right == null) {
      res.add(path.join('->'));
      return;
    }
    if (node.left != null) {
      backTrack(node.left!);
      path.removeLast();
    }
    if (node.right != null) {
      backTrack(node.right!);
      path.removeLast();
    }
  }

  backTrack(root);
  return res;
}

// 1/ 2 3/ 4 5 6 7
// 1 2 4 5 3 6 7
// 4 2 5  1  6 3 7
TreeNode? buildTree(List<int> preorder, List<int> inorder) {
  if (preorder.isEmpty || inorder.isEmpty) return null; // empty tree
  TreeNode root = TreeNode(preorder[0]);
  int rootIndex = inorder.indexOf(preorder[0]);
  root.left = buildTree(
      preorder.sublist(1, rootIndex + 1), inorder.sublist(0, rootIndex));
  root.right = buildTree(
      preorder.sublist(rootIndex + 1), inorder.sublist(rootIndex + 1));
  return root;
}

// 1/ 2 3/ 4 5 6 7
// 4 2 5 1 6 3 7
// 4 5 2 6 7 3 1
TreeNode? buildTree2(List<int> inorder, List<int> postorder) {
  if (inorder.isEmpty || postorder.isEmpty) return null; // empty tree
  TreeNode root = TreeNode(postorder.last);
  int rootIndex = inorder.indexOf(postorder.last);
  root.left = buildTree2(
      inorder.sublist(0, rootIndex), postorder.sublist(0, rootIndex));
  root.right = buildTree2(inorder.sublist(rootIndex + 1),
      postorder.sublist(rootIndex, postorder.length - 1));
  return root;
}
/**
 700. 二叉搜索树中的搜索
给定二叉搜索树（BST）的根节点 root 和一个整数值 val。

你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。
输入：root = [4,2,7,1,3], val = 2
输出：[2,1,3]
 */

TreeNode? searchBST(TreeNode? root, int val) {
  if (root == null) return null; // empty tree
  if (root.val == val) return root; // found the node
  if (val < root.val) {
    return searchBST(root.left, val); // search in left subtree
  } else {
    return searchBST(root.right, val); // search in right subtree
  }
}

List<List<int>> levelOrderBottom(TreeNode? root) {
  // List<List<int>> result = [];
  // if (root == null) return result; // empty tree
  // List<TreeNode> queue = [];
  // List<TreeNode> nextQueue = [];
  // List<int> currentLevel = [];
  // queue.insert(0, root);
  // while (queue.length > 0 || nextQueue.length > 0) {
  //   if (queue.length == 0) {
  //     queue = nextQueue;
  //     nextQueue = [];
  //     result.add(currentLevel); // insert at the beginning for bottom-up order
  //     currentLevel = [];
  //   }
  //   TreeNode? node = queue.removeLast();
  //   currentLevel.add(node.val); // visit the root
  //   if (node.left != null) {
  //     nextQueue.insert(0, node.left!); // add left child to next queue
  //   }
  //   if (node.right != null) {
  //     nextQueue.insert(0, node.right!); // add right child to next queue
  //   }
  // }
  // if (currentLevel.isNotEmpty) {
  //   result.add(currentLevel); // add the last level if not empty
  // }
  List<List<int>> result = [];
  if (root == null) return result; // empty tree
  List<TreeNode> queue = [];
  queue.insert(0, root);
  while (queue.isNotEmpty) {
    List<int> currentLevel = [];
    int size = queue.length;
    for (int i = 0; i < size; i++) {
      TreeNode node = queue.removeLast();
      currentLevel.add(node.val);
      if (node.left != null) queue.insert(0, node.left!);
      if (node.right != null) queue.insert(0, node.right!);
    }
    result.add(currentLevel);
  }
  return result.reversed.toList(); // reverse the result for bottom-up order
}
/**
 199. 二叉树的右视图
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
输入：root = [1,2,3,null,5,null,4]
输出：[1,3,4]
 */

List<int> rightSideView(TreeNode? root) {
  List<int> result = [];
  if (root == null) return result; // empty tree
  List<TreeNode> queue = [];
  queue.insert(0, root);
  while (queue.isNotEmpty) {
    List<int> currentLevel = [];
    int size = queue.length;
    for (int i = 0; i < size; i++) {
      TreeNode node = queue.removeLast();
      currentLevel.add(node.val);
      if (node.left != null) queue.insert(0, node.left!);
      if (node.right != null) queue.insert(0, node.right!);
    }
    result.add(currentLevel.last); // add the last node of the current level
  }
  return result;
}

/**
 给定一个非空二叉树的根节点 root , 以数组的形式返回每一层节点的平均值。与实际答案相差 10-5 以内的答案可以被接受。
输入：root = [3,9,20,null,null,15,7]
输出：[3.00000,14.50000,11.00000]
解释：第 0 层的平均值为 3,第 1 层的平均值为 14.5,第 2 层的平均值为 11 。
因此返回 [3, 14.5, 11] 。
 */
List<double> averageOfLevels(TreeNode? root) {
  List<double> result = [];
  if (root == null) return result; // empty tree
  List<TreeNode> queue = [];
  queue.insert(0, root);
  while (queue.isNotEmpty) {
    double sum = 0;
    int size = queue.length;
    for (int i = 0; i < size; i++) {
      TreeNode node = queue.removeLast();
      sum += node.val;
      if (node.left != null) queue.insert(0, node.left!);
      if (node.right != null) queue.insert(0, node.right!);
    }
    result.add(sum / size); // add the average of the current level
  }
  return result;
}

/**
 226. 翻转二叉树
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
 */

TreeNode? invertTree(TreeNode? root) {
  if (root == null) return null;
  void swapChildren(TreeNode? node) {
    if (node == null) return;
    TreeNode? temp = node.left;
    node.left = node.right;
    node.right = temp;
  }

  swapChildren(root);
  invertTree(root.left);
  invertTree(root.right);

  return root;
}

// -10,-3,0,5,9 10
TreeNode? sortedArrayToBST(List<int> nums) {
  if (nums.isEmpty) return null; // empty array
  int mid = nums.length ~/ 2; // find the middle index
  TreeNode root = TreeNode(nums[mid]); // create the root node
  root.left = sortedArrayToBST(nums.sublist(0, mid)); // left subtree
  root.right = sortedArrayToBST(nums.sublist(mid + 1)); // right subtree
  return root; // return the root of the BST
}

bool hasPathSum(TreeNode? root, int targetSum) {
  // if (root == null) return false; // empty tree, check if targetSum is 0
  // if (root.left == null && root.right == null) {
  //   return targetSum ==
  //       root.val; // leaf node, check if targetSum equals node value
  // }
  // return hasPathSum(root.left, targetSum - root.val) ||
  //     hasPathSum(
  //         root.right, targetSum - root.val); // check left and right subtrees

  // if (root == null) return false; // empty tree, check if targetSum is 0
  // int sum = 0;
  // bool backTrack(TreeNode node) {
  //   sum += node.val;
  //   if (node.left == null && node.right == null) {
  //     return sum == targetSum; // leaf node, check if remaining sum is 0
  //   }
  //   if (node.left != null) {
  //     if (backTrack(node.left!)) return true; // 如果在左子树找到了路径，尽快返回true，不需要再继续了
  //     sum -= node.left!.val; // 没有找到，回溯，继续从右子树找
  //   }
  //   if (node.right != null) {
  //     if (backTrack(node.right!)) return true;
  //     sum -= node.right!.val; // 没有找到，回溯，继续从（右 右）子树找
  //   }
  //   return false;
  // }

  // return backTrack(root);

  if (root == null) return false; // empty tree, check if targetSum is 0
  int sum = 0;
  bool backTrack(TreeNode? node) {
    if (node == null) return false;
    sum += node.val;
    if (node.left == null && node.right == null) {
      if (sum == targetSum)
        return true; // 只有true才返回，false还需要回溯 sum -= node.val;
    }
    if (backTrack(node.left)) return true; // 如果在左子树找到了路径，尽快返回true，不需要再继续了
    if (backTrack(node.right)) return true;
    sum -= node.val; // 没有找到，回溯，继续从（右 右）子树找
    return false;
  }

  return backTrack(root);
}

List<List<int>> pathSum(TreeNode? root, int targetSum) {
  List<List<int>> result = [];
  List<int> currentPath = [];
  if (root == null) return result; // empty tree, return empty result

  void backTrack(TreeNode? node, int currentSum) {
    if (node == null) return; // base case: empty node
    currentPath.add(node.val); // add the leaf node value
    final int remaining = currentSum - node.val;
    if (node.left == null && node.right == null) {
      if (remaining == 0) {
        result.add(List.from(currentPath)); // add the current path to result
      }
      currentPath.removeLast(); // backtrack, remove the leaf node value
      return;
    }

    backTrack(node.left, remaining); // start backtracking from the root
    backTrack(node.right, remaining); // check right subtree
    currentPath.removeLast(); // remove the root value from the current path
  }

  backTrack(root, targetSum); // start backtracking from the root
  return result;
}

// 输入：root = [1,2,5,3,4,null,6]
// 输出：[1,null,2,null,3,null,4,null,5,null,6]
void flatten(TreeNode? root) {
  if (root == null) return; // empty tree, nothing to flatten
  List<TreeNode> stack = [];
  List<TreeNode> result = [];
  stack.add(root);
  while (stack.length > 0) {
    TreeNode? node = stack.removeLast();
    result.add(node); // visit the root
    if (node.right != null) {
      stack.add(node.right!); // add right child to stack
    }
    if (node.left != null) {
      stack.add(node.left!); // add left child to stack
    }
  }
  for (int i = 0; i < result.length - 1; i++) {
    result[i].left = null; // set left child to null
    result[i].right = result[i + 1]; // link to the next node
  }
}

// 输入: numRows = 5
// 输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
List<List<int>> generate(int numRows) {
  List<List<int>> result = [];

  for (int i = 1; i <= numRows; i++) {
    List<int> row = List.filled(i, 0);
    for (int j = 0; j < i; j++) {
      if (j == 0 || j == i - 1)
        row[j] = 1;
      else {
        row[j] = result[i - 2][j - 1] + result[i - 2][j];
      }
    }
    result.add(row); // add the current row to the result
  }
  return result;
}

List<int> getRow(int rowIndex) {
  List<int> lastRow = List.filled(rowIndex, 0);
  List<int> currentRow = List.filled(rowIndex + 1, 0);
  for (int i = 0; i <= rowIndex; i++) {
    for (int j = 0; j <= i; j++) {
      if (j == 0 || j == i) {
        currentRow[j] = 1; // first and last element of the row is always 1
      } else {
        currentRow[j] =
            lastRow[j - 1] + lastRow[j]; // sum of the two elements above
      }
    }
    lastRow = List.from(
        currentRow); // update lastRow to currentRow for the next iteration
  }
  return currentRow;
}
//0    2
//1   3 4
//2  6 5 7
//3 4 1 8 3

int minimumTotal(List<List<int>> triangle) {
  for (int i = 1; i < triangle.length; i++) {
    for (int j = 0; j <= i; j++) {
      if (j == 0) {
        triangle[i][j] +=
            triangle[i - 1][0]; // first and last element of the row
      } else if (j == i) {
        triangle[i][j] +=
            triangle[i - 1][i - 1]; // last element of the row is always 1
      } else {
        triangle[i][j] += min(triangle[i - 1][j - 1],
            triangle[i - 1][j]); // sum of the two elements above
      }
    }
  }
  int minNum = triangle[triangle.length - 1][0];
  for (int i = 0; i < triangle.length; i++) {
    minNum = min(minNum, triangle[triangle.length - 1][i]);
  }
  return minNum; // return the minimum path sum
}

/**
 121. 买卖股票的最佳时机, 只交易一笔
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
输入：[7,1,5,3,6,4]
输出：5
【10，17， 2，10】
 */
// [7,1,5,3,6,4]
// [2,1,2,0,1]
// [2,1,2,1,0,1,2]
int maxProfit(List<int> prices) {
  // //思路：区域最小值和最大值
  // int minPrice = 100001;
  // int maxProfit = 0;
  // for (int i = 0; i < prices.length; i++) {
  //   if (prices[i] < minPrice) minPrice = prices[i]; // update min price
  //   if (prices[i] - minPrice > maxProfit)
  //     maxProfit = prices[i] - minPrice; // update max profit
  // }
  // return maxProfit; // return the maximum profit

  //思路：动态规划，dp[i][0]记录第i天持有股票后手中金钱最大值, dp[i][1]记录第i天不持有股票后手中金钱最大值
  List<List<int>> dp =
      List.generate(prices.length, (index) => List.filled(2, 0));
  dp[0][0] = -prices[0];
  dp[0][1] = 0;
  for (int i = 1; i < prices.length; i++) {
    // 因为只能买卖一次，所以如果决定今天买入，而且初始金钱是0，所以是 0 - prices[i]
    dp[i][0] = max(dp[i - 1][0], -prices[i]);
    dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0]);
  }
  return dp[prices.length - 1][1];
}

/**
 122. 买卖股票的最佳时机 II，可以多次交易
给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。
 */
int maxProfit2(List<int> prices) {
  // int maxProfit = 0;
  // for (int i = 1; i < prices.length; i++) {
  //   if (prices[i] > prices[i - 1]) {
  //     maxProfit += prices[i] - prices[i - 1]; // add profit if price increases
  //   }
  // }
  // return maxProfit; // return the maximum profit

  List<List<int>> dp =
      List.generate(prices.length, (index) => List.filled(2, 0));
  dp[0][0] = -prices[0];
  dp[0][1] = 0;
  for (int i = 1; i < prices.length; i++) {
    // 因为可以交易多次，所以如果决定今天买入，那当前的金钱是昨天卖出的利润减去今天买入的价格 dp[i - 1][1] - prices[i]
    dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
    dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
  }
  return dp[prices.length - 1][1];
}

/**
 123. 买卖股票的最佳时机 III，最多可以完成两笔交易

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
 */
// prices = [1,2,3,4,5]
// 输入：prices = [3,3,5,0,0,3,1,4]
// 输出：6
// 解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
//      随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
//  int n = prices.length;
//         int buy1 = -prices[0], sell1 = 0;
//         int buy2 = -prices[0], sell2 = 0;
//         for (int i = 1; i < n; ++i) {
//             buy1 = Math.max(buy1, -prices[i]);
//             sell1 = Math.max(sell1, buy1 + prices[i]);
//             buy2 = Math.max(buy2, sell1 - prices[i]);
//             sell2 = Math.max(sell2, buy2 + prices[i]);
//         }
//         return sell2;
int maxProfit3(List<int> prices) {
  // 这个写法是错误的，无法处理【100，101，0，99， 1， 100】这种情况
  // int left = 0, right = 0;
  // int sum = 0;
  // int maxProfit(List<int> prices) {
  //   int res = 0;
  //   int minPriceIndex = 0;
  //   for (int i = 1; i < prices.length; i++) {
  //     if (prices[i] < prices[minPriceIndex]) {
  //       minPriceIndex = i;
  //     }
  //     if (prices[i] - prices[minPriceIndex] > res) {
  //       res = prices[i] - prices[minPriceIndex];
  //       left = minPriceIndex;
  //       right = i;
  //     }
  //   }
  //   return res;
  // }
  // sum += maxProfit(prices);
  // int leftMax = maxProfit(  prices.sublist(0, left));
  // int rightMax = maxProfit(prices.sublist(right + 1));
  // sum += max(leftMax, rightMax);
  // return sum;

//【100，101，0，99， 2，1，2， 100】
  // int buy1 = -prices[0], sell1 = 0;
  // int buy2 = -prices[0], sell2 = 0;
  // //总的思路就是买入花钱，卖出赚钱
  // //第二次买入的时候需要加上第一次卖出的利润
  // for (int i = 1; i < prices.length; i++) {
  //   buy1 = max(buy1, -prices[i]);
  //   sell1 = max(sell1, buy1 + prices[i]);
  //   buy2 = max(buy2, sell1 - prices[i]);
  //   sell2 = max(sell2, buy2 + prices[i]);
  // }
  // return sell2; // return the maximum profit from two transactions

  List<List<int>> dp =
      List.generate(prices.length, (index) => List.filled(5, 0));
  dp[0][1] = -prices[0];
  dp[0][3] = -prices[0];
  for (int i = 1; i < prices.length; i++) {
    dp[i][1] = max(dp[i - 1][1], -prices[i]);
    dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
    dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
    dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
  }

  return dp[prices.length - 1][4];
}

// [1,-2,-3,1,3,-2,null,-1]
// yu 3 shuchu 4
int maxPathSum(TreeNode? root) {
  int maxSum = -10001; // minimum value for int

  int backTrack(TreeNode? node) {
    if (node == null) return -10001; // empty tree has no path sum
    int leftMax = backTrack(node.left); // path sum of left subtree
    int rightMax = backTrack(node.right); // path sum of right subtree

    int currentMax = node.val;
    currentMax = max(currentMax, node.val + leftMax); // 连接左子树
    currentMax = max(currentMax, node.val + rightMax); // 连接右子树

    int maxWithBoth = node.val;
    if (leftMax > -10001) maxWithBoth += leftMax;
    if (rightMax > -10001) maxWithBoth += rightMax;
    maxSum = max(maxSum, maxWithBoth); // 更新全局和
    maxSum = max(maxSum, currentMax); // 当前节点的单路径

    return currentMax;
  }

  backTrack(root);
  return maxSum;
}

bool isPalindrome2(String s) {
  int left = 0, right = s.length - 1;
  while (left <= right) {
    if (!s[left].contains(RegExp(r'[a-zA-Z0-9]'))) {
      left++; // skip non-alphanumeric characters
      continue;
    }
    if (!s[right].contains(RegExp(r'[a-zA-Z0-9]'))) {
      right--; // skip non-alphanumeric characters
      continue;
    }
    if (s[left].toLowerCase() != s[right].toLowerCase()) {
      return false; // characters do not match
    }
    left++;
    right--;
  }
  return true;
}

int longestConsecutive(List<int> nums) {
  if (nums.isEmpty) return 0;

  // 使用 Set 去重
  final numSet = Set<int>.from(nums);
  int result = 0;

  // 直接遍历集合而非原始数组
  for (final num in numSet) {
    // 仅当元素是连续序列起点时处理
    if (!numSet.contains(num - 1)) {
      int currentNum = num;
      int currentStreak = 1; // 直接从1开始计数

      // 检查连续递增序列
      while (numSet.contains(currentNum + 1)) {
        currentNum++;
        currentStreak++;
      }

      result = max(result, currentStreak);
    }
  }

  return result;
}

//  4
// 9 0
//5 1
int sumNumbers(TreeNode? root) {
  int backTrack(TreeNode? node, int sum) {
    if (node == null) return 0;
    int newSum = sum * 10 + node.val;
    if (node.left == null && node.right == null) return newSum;
    int left = backTrack(node.left, newSum);
    int right = backTrack(node.right, newSum);
    return left + right;
  }

  return backTrack(root, 0);
}

void solve(List<List<String>> board) {
  void backTrack(int i, int j) {
    // 从边界上找连起来的'O'，并将其标记为'A'
    if (i < 0 ||
        i >= board.length ||
        j < 0 ||
        j >= board[0].length ||
        board[i][j] != 'O') return;
    board[i][j] = 'A'; // mark as visited
    backTrack(i + 1, j);
    backTrack(i - 1, j);
    backTrack(i, j + 1);
    backTrack(i, j - 1);
  }

  for (int i = 0; i < board.length; i++) {
    backTrack(i, 0);
    backTrack(i, board[i].length - 1);
  }
  for (int j = 1; j < board[0].length - 1; j++) {
    backTrack(0, j);
    backTrack(board.length - 1, j);
  }
  for (int i = 0; i < board.length; i++) {
    for (int j = 0; j < board[0].length; j++) {
      if (board[i][j] == 'A') {
        board[i][j] = 'O'; // restore visited cells to 'O'
      } else {
        board[i][j] = 'X'; // change 'O' to 'X'
      }
    }
  }
}

int minCut(String s) {
  int minCuts = s.length - 1; // maximum cuts possible
  List<String> current = [];
  bool isPalindrome(String str) {
    int left = 0, right = str.length - 1;
    while (left < right) {
      if (str[left] != str[right]) return false; // characters do not match
      left++;
      right--;
    }
    return true; // all characters match
  }

  void backTrack(int start) {
    if (start == s.length) {
      minCuts = min(minCuts, current.length - 1); // update minimum cuts
      return;
    }
    for (int i = start; i < s.length; i++) {
      String substring = s.substring(start, i + 1);
      if (current.length - 1 >= minCuts) return;
      if (isPalindrome(substring)) {
        current.add(substring); // add valid palindrome substring
        backTrack(i + 1); // move to the next index
        current.removeLast(); // backtrack
      }
    }
  }

  backTrack(0); // start backtracking from index 0
  return minCuts;
}

/**
 134. 加油站
在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明: 

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。
示例 1: 输入:

gas = [1,2,3,4,5]
cost = [3,4,5,1,2]
 */
// 输入: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
// 输出: 3
int canCompleteCircuit(List<int> gas, List<int> cost) {
  int start = 0;
  int totalGas = 0, currentGas = 0;
  for (int i = 0; i < gas.length; i++) {
    currentGas += gas[i] - cost[i]; // calculate net gas at each station
    totalGas += gas[i] - cost[i]; // calculate total net gas
    if (currentGas < 0) {
      start = i + 1; // reset start index if current gas is negative
      currentGas = 0; // reset current gas
    }
  }
  if (totalGas < 0 || start >= gas.length) {
    return -1; // not enough gas to complete the circuit
  }
  return start; // return the starting index
}

// 1 2 2 5 4 3 2
// 1 2 1 2 1 1 1
// 1 1 1 4 3 2 1
/**
 135. 分发糖果
力扣题目链接

老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
相邻的孩子中，评分高的孩子必须获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？

示例 1:

输入: [1,0,2]
输出: 5
解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
 */
int candy(List<int> ratings) {
  List<int> candisLeft = List.filled(ratings.length, 1);
  List<int> candisRight = List.filled(ratings.length, 1);
  for (int i = 1; i < ratings.length; i++) {
    if (ratings[i] > ratings[i - 1]) {
      candisLeft[i] = candisLeft[i - 1] + 1; // left to right
    }
  }
  for (int i = ratings.length - 2; i >= 0; i--) {
    if (ratings[i] > ratings[i + 1]) {
      candisRight[i] = candisRight[i + 1] + 1; // right to left
    }
  }

  int totalCandies = 0;
  for (int i = 0; i < ratings.length; i++) {
    totalCandies += max(candisLeft[i], candisRight[i]); // take the maximum
  }
  return totalCandies; // return the total candies needed
}

/**
 860.柠檬水找零
在柠檬水摊上，每一杯柠檬水的售价为 5 美元。

顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。

每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。

注意，一开始你手头没有任何零钱。

如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

示例 1：

输入：[5,5,5,10,20]
输出：true
提示：
bills[i] 不是 5 就是 10 或是 20 
 */
bool lemonadeChange(List<int> bills) {
  int five = 0;
  int ten = 0;
  for (int i = 0; i < bills.length; i++) {
    if (bills[i] == 5) {
      five++;
    } else if (bills[i] == 10) {
      if (five == 0) return false;
      five--;
      ten++;
    } else {
      if (ten > 0 && five > 0) {
        ten--;
        five--;
      } else if (five >= 3) {
        five -= 3;
      } else {
        return false;
      }
    }
  }
  return true;
}
/**
 406. 根据身高重建队列
假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，
前面 正好 有 ki 个身高大于或等于 hi 的人。

请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
 */
// 70, 50, 71, 61, 52, 44,
//50, 71, 61, 52, 44

List<List<int>> reconstructQueue(List<List<int>> people) {
  people.sort((a, b) {
    if (a[0] != b[0]) {
      return b[0] - a[0]; // sort by height in descending order
    } else {
      return a[1] - b[1]; // if heights are equal, sort by k in ascending order
    }
  });
  List<List<int>> res = [];
  for (var person in people) {
    res.insert(person[1], person); // insert person at index k
  }
  return res; // return the reconstructed queue
}
/**
 435. 无重叠区间
给定一个区间的集合 intervals ，其中 intervals[i] = [starti, endi] 。返回 需要移除区间的最小数量，使剩余区间互不重叠 。

注意 只在一点上接触的区间是 不重叠的。例如 [1, 2] 和 [2, 3] 是不重叠的。

 

示例 1:

输入: intervals = [[1,2],[2,3],[3,4],[1,3]]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。
 */

int eraseOverlapIntervals(List<List<int>> intervals) {
  int res = 0;
  intervals.sort((a, b) {
    if (a[0] != b[0]) return a[0] - b[0];
    return a[1] - b[1];
  });
  print(intervals);
  int end = intervals[0][1];
  for (int i = 1; i < intervals.length; i++) {
    if (intervals[i][0] < end) {
      res++;
      end = min(end, intervals[i][1]);
    } else
      end = intervals[i][1];
  }
  return res;
}

int singleNumber(List<int> nums) {
  int result = 0;
  for (int num in nums) {
    result ^= num; // XOR operation to find the single number
  }
  return result; // return the single number
}

int singleNumber2(List<int> nums) {
  Map<int, int> countMap = {};
  for (int num in nums) {
    if (countMap.containsKey(num)) {
      countMap[num] = countMap[num]! + 1; // increment the count
    } else {
      countMap[num] = 1; // initialize the count
    }
  }
  for (int num in countMap.keys) {
    if (countMap[num] == 1) {
      return num; // return the number that appears only once
    }
  }
  return -1; // if no single number found, return -1
}
/* 
给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。
 */

bool wordBreak(String s, List<String> wordDict) {
  List<bool> dp = List.filled(s.length + 1, false);
  dp[0] = true; // base case: empty string can be formed
  Set<String> wordSet =
      Set.from(wordDict); // convert list to set for faster lookup
  for (int i = 1; i <= s.length; i++) {
    for (int j = 0; j < i; j++) {
      if (dp[j] && wordSet.contains(s.substring(j, i))) {
        dp[i] = true; // if substring can be formed, mark as true
      }
    }
  }
  return dp[s.length]; // return true if the whole string can be formed
}

List<String> wordBreak2(String s, List<String> wordDict) {
  List<String> result = [];
  List<String> current = [];
  void backTrack(int index) {
    if (index == s.length) {
      result.add(current.join(" ")); // add the current partition to result
      return;
    }
    for (int i = index; i < s.length; i++) {
      String word = s.substring(index, i + 1);
      if (wordDict.contains(word)) {
        current.add(word); // add valid word to current partition
        backTrack(i + 1); // move to the next index
        current.removeLast(); // backtrack
      }
    }
  }

  backTrack(0); // start backtracking from index 0

  return result;
}

/**
输入:s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
输出:["cats and dog","cat sand dog"]
示例 2：

输入:s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
输出:["pine apple pen apple","pineapple pen apple","pine applepen apple"]
解释: 注意你可以重复使用字典中的单词。


对于字符串 s，如果某个前缀是单词列表中的单词，则拆分出该单词，然后对 s 的剩余部分继续拆分。如果可以将整个字符串 s 拆分成单词列表中的单词，则得到一个句子。
在对 s 的剩余部分拆分得到一个句子之后，将拆分出的第一个单词（即 s 的前缀）添加到句子的头部，即可得到一个完整的句子。上述过程可以通过回溯实现。

假设字符串 s 的长度为 n，回溯的时间复杂度在最坏情况下高达 O(n 
n
 )。时间复杂度高的原因是存在大量重复计算，可以通过记忆化的方式降低时间复杂度。

具体做法是，使用哈希表存储字符串 s 的每个下标和从该下标开始的部分可以组成的句子列表，在回溯过程中如果遇到已经访问过的下标，则可以直接从哈希表得到结果，而不需要重复计算。
如果到某个下标发现无法匹配，则哈希表中该下标对应的是空列表，因此可以对不能拆分的情况进行剪枝优化。

还有一个可优化之处为使用哈希集合存储单词列表中的单词，这样在判断一个字符串是否是单词列表中的单词时只需要判断该字符串是否在哈希集合中即可，而不再需要遍历单词列表。


class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        Map<Integer, List<List<String>>> map = new HashMap<Integer, List<List<String>>>();
        List<List<String>> wordBreaks = backtrack(s, s.length(), new HashSet<String>(wordDict), 0, map);
        List<String> breakList = new LinkedList<String>();
        for (List<String> wordBreak : wordBreaks) {
            breakList.add(String.join(" ", wordBreak));
        }
        return breakList;
    }

    public List<List<String>> backtrack(String s, int length, Set<String> wordSet, int index, Map<Integer, List<List<String>>> map) {
        if (!map.containsKey(index)) {
            List<List<String>> wordBreaks = new LinkedList<List<String>>();
            if (index == length) {
                wordBreaks.add(new LinkedList<String>());
            }
            for (int i = index + 1; i <= length; i++) {
                String word = s.substring(index, i);
                if (wordSet.contains(word)) {
                    List<List<String>> nextWordBreaks = backtrack(s, length, wordSet, i, map);
                    for (List<String> nextWordBreak : nextWordBreaks) {
                        LinkedList<String> wordBreak = new LinkedList<String>(nextWordBreak);
                        wordBreak.offerFirst(word);
                        wordBreaks.add(wordBreak);
                    }
                }
            }
            map.put(index, wordBreaks);
        }
        return map.get(index);
    }
}

 */
//  1 2 3 4
// 输入：head = [1,2,3,4, 5]
// 输出：[1,5,2,4,3]
// 输入：head = [1,2,3,4,5,6]
// 输出：[1,6,2,5,3,4]
// 1 2 3  6 5 4
void reorderList(ListNode? head) {
  if (head == null || head.next == null) return; // empty or single node list
  ListNode? dummy = ListNode(0, head);

  ListNode getMid(ListNode head) {
    ListNode? slow = head;
    ListNode? fast = head;
    while (fast != null && fast.next != null) {
      slow = slow!.next; // slow moves one step
      fast = fast.next!.next; // fast moves two steps
    }
    return slow!; // return the middle node
  }

  ListNode? reverseNode(ListNode? head) {
    ListNode? prev = null, cur = head;
    while (cur != null) {
      ListNode? next = cur.next; // save the next node
      cur.next = prev; // reverse the link
      prev = cur; // move prev to current node
      cur = next; // move to the next node
    }
    return prev; // return the new head of the reversed list
  }

  void mergeNode(ListNode? l1, ListNode? l2) {
    while (l1 != null && l2 != null) {
      ListNode? next1 = l1.next; // save the next node of l1
      ListNode? next2 = l2.next; // save the next node of l2
      l1.next = l2; // link l1 to l2
      l2.next = next1; // link l2 to the next node of l1
      l1 = next1; // move l1 to the next node
      l2 = next2; // move l2 to the next node
    }
  }

  ListNode mid = getMid(head); // find the middle node

  ListNode? l2 = mid.next; // second half starts from mid.next
  mid.next = null; // split the list into two halves
  l2 = reverseNode(l2); // reverse the second half of the list
  mergeNode(dummy.next, l2);
}

//  1 2 3
// 1 2 3
// 1 3 2
List<int> preorderTraversal3(TreeNode? root) {
  List<int> result = [];
  List<TreeNode?> stackIn = [];
  List<TreeNode?> stackOut = [];
  if (root == null) return result; // empty tree, return empty result
  stackIn.add(root); // start with the root node
  while (stackIn.length > 0) {
    TreeNode? node = stackIn.removeLast(); // get the last node from the stack
    if (node != null) {
      stackOut.add(node); // visit the root

      if (node.left != null) {
        stackIn.add(node.left); // add left child to stack
      }
      if (node.right != null) {
        stackIn.add(node.right); // add right child to stack
      }
    }
  }
  while (stackOut.length > 0) {
    TreeNode? node =
        stackOut.removeLast(); // get the last node from the output stack
    if (node != null) {
      result.add(node.val); // add the node value to the result
    }
  }
  return result;
}
/**
 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，
则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

 

示例：

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4


LRU 缓存机制可以通过哈希表辅以双向链表实现，我们用一个哈希表和一个双向链表维护所有在缓存中的键值对。

双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。

哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。

这样以来，我们首先使用哈希表进行定位，找出缓存项在双向链表中的位置，随后将其移动到双向链表的头部，即可在 O(1) 的时间内完成 get 或者 put 操作。具体的方法如下：

对于 get 操作，首先判断 key 是否存在：

如果 key 不存在，则返回 −1；

如果 key 存在，则 key 对应的节点是最近被使用的节点。通过哈希表定位到该节点在双向链表中的位置，并将其移动到双向链表的头部，最后返回该节点的值。

对于 put 操作，首先判断 key 是否存在：

如果 key 不存在，使用 key 和 value 创建一个新的节点，在双向链表的头部添加该节点，并将 key 和该节点添加进哈希表中。
然后判断双向链表的节点数是否超出容量，如果超出容量，则删除双向链表的尾部节点，并删除哈希表中对应的项；

如果 key 存在，则与 get 操作类似，先通过哈希表定位，再将对应的节点的值更新为 value，并将该节点移到双向链表的头部。

上述各项操作中，访问哈希表的时间复杂度为 O(1)，在双向链表的头部添加节点、在双向链表的尾部删除节点的复杂度也为 O(1)。
而将一个节点移到双向链表的头部，可以分成「删除该节点」和「在双向链表的头部添加节点」两步操作，都可以在 O(1) 时间内完成。


 */

class TowListNode {
  int key;
  int value;
  TowListNode? next;
  TowListNode? prev;
  TowListNode([this.key = 0, this.value = 0, this.next, this.prev]);
}

class LRUCache {
  int size = 0;
  Map<int, TowListNode> cache = {};
  int capacity = 0;
  TowListNode? head; // create a dummy head node
  TowListNode? tail; // create a dummy tail node

  LRUCache(int capacity) {
    size = 0;
    this.capacity = capacity; // initialize the cache with given capacity

    // 方便向头部添加，和尾部删除
    head = TowListNode(0, 0); // create a dummy head node
    tail = TowListNode(0, 0); // create a dummy tail node
    head?.next = tail; // link head to tail
    tail?.prev = head; // link tail to head
    cache = {}; // initialize the cache as an empty map
  }

  int get(int key) {
    TowListNode? node = cache[key];
    if (node == null) {
      return -1; // key does not exist, return -1
    } else {
      moveToHead(node);
      return node.value; // return the value of the node
    }
  }

  void put(int key, int value) {
    TowListNode? node = cache[key];
    if (node == null) {
      TowListNode newNode = TowListNode(key, value);
      cache[key] = newNode; // add the new node to the cache
      addToHead(newNode);
      size++; // increment the size of the cache
      if (size > capacity) {
        TowListNode tailPre = removeTail();
        cache.remove(tailPre.key); // remove the least recently used node
        size--; // decrement the size of the cache
      } else {
        cache[key] = newNode; // add the new node to the cache
      }
    } else {
      // key exists, update the value and move to front
      node.value = value;
      // move node to front logic here
      moveToHead(node);
    }
  }

  void removeNode(TowListNode node) {
    node.next?.prev = node.prev;
    node.prev?.next = node.next; // unlink the node from the list
  }

  void addToHead(TowListNode node) {
    TowListNode? headNext = head?.next; // get the current first node
    node.prev = head;
    node.next = headNext; // link the new node to the tail
    head?.next = node;
    headNext?.prev = node; // link the current first node to the new node
  }

  TowListNode removeTail() {
    TowListNode? deleteNode = tail?.prev; // get the last node
    removeNode(deleteNode!);
    return deleteNode;
  }

  void moveToHead(TowListNode node) {
    removeNode(node); // remove the node from its current position
    addToHead(node); // add the node to the head of the list
  }
}

/**
 给定单个链表的头 head ，使用 插入排序 对链表进行排序，并返回 排序后链表的头 。

插入排序 算法的步骤:

插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
重复直到所有输入数据插入完为止。
下面是插入排序算法的一个图形示例。部分排序的列表(黑色)最初只包含列表中的第一个元素。每次迭代时，从输入数据中删除一个元素(红色)，并就地插入已排序的列表中。

对链表进行插入排序。
 */

// 4
// 1 3 5 4
// 1 2 3 4 10 5 6 7 8
ListNode? insertionSortList(ListNode? head) {
  if (head == null || head.next == null) {
    return head; // empty or single node list, return as is
  }
  ListNode dummy = ListNode(-5001, head); // create a dummy head node

  ListNode? deleteNode(ListNode node) {
    if (node.next == null) return null;
    node.next = node.next!.next; // unlink the next node
    return node.next!; // return the next node to delete
  }

  ListNode? current = head; // start with the first node
  ListNode? pre = dummy;
  while (current != null && pre != null) {
    if (current.val >= pre.val) {
      current =
          current.next; // move to the next node if current is greater than pre
      pre = pre.next; // move pre to the next node
    } else {
      ListNode? nextCurrent = current.next; // save the next node
      ListNode? insertPre = dummy; // start from the dummy head
      while (insertPre != pre && insertPre != null && insertPre.next != null) {
        if (insertPre.next!.val > current.val) {
          current.next = insertPre.next; // insert current after insertPre
          insertPre.next = current;
          break;
        } else {
          insertPre = insertPre.next; // move to the next node
        }
      }
      pre.next = nextCurrent;
      current = nextCurrent; // move to the next node
    }
  }
  return dummy.next; // return the sorted list starting from the dummy head
}

ListNode? sortList(ListNode? head) {
  ListNode merge(ListNode? l1, ListNode? l2) {
    ListNode dummy = ListNode(0);
    ListNode current = dummy;
    while (l1 != null && l2 != null) {
      if (l1.val < l2.val) {
        current.next = l1;
        l1 = l1.next;
      } else {
        current.next = l2;
        l2 = l2.next;
      }
      current = current.next!;
    }
    current.next = l1 ?? l2; // append the remaining nodes
    return dummy.next!; // return the merged list
  }

  // 0 1 2 3 4 5
  ListNode? sortNodeList(ListNode? head) {
    if (head == null || head.next == null) {
      return head;
    }
    ListNode dummy = ListNode(0, head);
    ListNode? slow = dummy;
    ListNode? fast = dummy;
    while (fast != null && fast.next != null) {
      slow = slow!.next; // slow moves one step
      fast = fast.next!.next; // fast moves two steps
    }
    ListNode? mid = slow!.next; // mid is the next node of slow
    slow.next = null; // split the list into two halves
    ListNode? left = sortNodeList(dummy.next); // sort the left half
    ListNode? right = sortNodeList(mid); // sort the right half
    return merge(left, right); // merge the two sorted halves
  }

  return sortNodeList(head); // start sorting from the head
}

/**
 给你一个数组 points ，其中 points[i] = [xi, yi] 表示 X-Y 平面上的一个点。求最多有多少个点在同一条直线上。

 

示例 1：


输入：points = [[1,1],[2,2],[3,3]]
输出：3
 */
int maxPoints(List<List<int>> points) {
  if (points.length <= 2)
    return points.length; // if there are 2 or fewer points, return the count

  int maxCount = 0;

  for (int i = 0; i < points.length; i++) {
    Map<double, int> slopeCount = {};
    for (int j = i + 1; j < points.length; j++) {
      int x = points[j][0] - points[i][0];
      int y = points[j][1] - points[i][1];
      double slop = 0;
      if (x == 0)
        slop = double.infinity;
      else if (y == 0)
        slop = 0;
      else
        slop = y / x; // calculate the slope
      if (slopeCount.containsKey(slop)) {
        slopeCount[slop] = slopeCount[slop]! + 1;
      } else {
        slopeCount[slop] = 1; // initialize the count for this slope
      }
      maxCount = max(maxCount, slopeCount[slop]!); // update max count
    }
  }
  return maxCount + 1; // return the maximum number of points on a line
}
/*
150. 逆波兰表达式求值
给你一个字符串数组 tokens ，表示一个根据 逆波兰表示法 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

注意：

有效的算符为 '+'、'-'、'*' 和 '/' 。
每个操作数（运算对象）都可以是一个整数或者另一个表达式。
两个整数之间的除法总是 向零截断 。
示例 1：

输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
 */

int evalRPN(List<String> tokens) {
  List<int> stack = [];
  for (int i = 0; i < tokens.length; i++) {
    if (tokens[i] == '*') {
      int b = stack.removeLast(); // pop the last two operands
      int a = stack.removeLast();
      stack.add(a * b); // push the result of multiplication
    } else if (tokens[i] == '/') {
      int b = stack.removeLast(); // pop the last two operands
      int a = stack.removeLast();
      stack.add(a ~/ b); // push the result of division, truncating towards zero
    } else if (tokens[i] == '+') {
      int b = stack.removeLast(); // pop the last two operands
      int a = stack.removeLast();
      stack.add(a + b); // push the result of addition
    } else if (tokens[i] == '-') {
      int b = stack.removeLast(); // pop the last two operands
      int a = stack.removeLast();
      stack.add(a - b); // push the result of subtraction
    } else {
      stack.add(int.parse(tokens[i])); // push the operand to the stack
    }
  }
  return stack.last; // return the last element in the stack as the result
}
/**
 239. 滑动窗口最大值
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。

输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
 */

List<int> maxSlidingWindow(List<int> nums, int k) {
  /**
   思路：对于窗口里的元素{2, 3, 5, 1 ,4}，单调队列里只维护（出）{5, 4}（进） 就够了，保持单调队列里单调递减，此时队列出口元素就是窗口里最大元素。
pop(value)：如果窗口移除的元素value等于单调队列的出口元素，那么队列弹出元素，否则不用任何操作
push(value)：如果push的元素value大于入口元素的数值，那么就将队列入口的元素弹出，再把当前元素压入，直到push元素的数值小于等于队列入口元素的数值为止
   */
  // 0 1  2  3 4 5 6 7
  // 1,3,-1,-3,5,3,6,7
  if (k > nums.length) return [];
  List<int> result = [];
  // List<int> deque = []; // will store indices of array elements
  DoubleLinkedQueue<int> deque = DoubleLinkedQueue<int>(); // 存储索引
  for (int i = 0; i < k; i++) {
    while (deque.isNotEmpty && deque.last < nums[i]) {
      deque.removeLast();
    }
    deque.addLast(nums[i]);
  }
  result.add(deque.first);
  for (int i = k; i < nums.length; i++) {
    if (nums[i - k] == deque.first) {
      deque.removeFirst();
    }

    while (deque.isNotEmpty && deque.last < nums[i]) {
      deque.removeLast();
    }
    deque.addLast(nums[i]);

    result.add(deque.first);
  }
  return result;
}

// 数组 / 字符串
// 合并两个有序数组
// 题解
// 简单

// # 88数组双指针排序
// 移除元素
// 题解
// 简单

// # 27数组双指针
// 删除有序数组中的重复项
// 题解
// 简单

// # 26数组双指针
// 删除有序数组中的重复项 II
// 题解
// 中等

// # 80数组双指针
// 多数元素
// 题解
// 简单

// # 169数组哈希表分治2+
// 轮转数组
// 题解
// 中等

// # 189数组数学双指针
// 买卖股票的最佳时机
// 题解
// 简单

// # 121数组动态规划
// 买卖股票的最佳时机 II
// 题解
// 中等

// # 122贪心数组动态规划
// 跳跃游戏
// 题解
// 中等

// # 55贪心数组动态规划
// 跳跃游戏 II
// 题解
// 中等

// # 45贪心数组动态规划
// H 指数
// 题解
// 中等

// # 274数组计数排序排序
// O(1) 时间插入、删除和获取随机元素
// 题解
// 中等

// # 380设计数组哈希表2+
// 除自身以外数组的乘积
// 题解
// 中等

// # 238数组前缀和
// 加油站
// 题解
// 中等

// # 134贪心数组
// 分发糖果
// 题解
// 困难

// # 135贪心数组
// 接雨水
// 题解
// 困难

// # 42栈数组双指针2+
// 罗马数字转整数
// 题解
// 简单

// # 13哈希表数学字符串
// 整数转罗马数字
// 题解
// 中等

// # 12哈希表数学字符串
// 最后一个单词的长度
// 题解
// 简单

// # 58字符串
// 最长公共前缀
// 题解
// 简单

// # 14字典树字符串
// 反转字符串中的单词
// 题解
// 中等

// # 151双指针字符串
// Z 字形变换
// 题解
// 中等

// # 6字符串
// 找出字符串中第一个匹配项的下标
// 题解
// 简单

// # 28双指针字符串字符串匹配
// 文本左右对齐
// 题解
// 困难

// # 68数组字符串模拟
// 双指针
// 验证回文串
// 题解
// 简单

// # 125双指针字符串
// 判断子序列
// 题解
// 简单

// # 392双指针字符串动态规划
// 两数之和 II - 输入有序数组
// 题解
// 中等

// # 167数组双指针二分查找
// 盛最多水的容器
// 题解
// 中等

// # 11贪心数组双指针
// 三数之和
// 题解
// 中等

// # 15数组双指针排序
// 滑动窗口
// 长度最小的子数组
// 题解
// 中等

// # 209数组二分查找前缀和1+
// 无重复字符的最长子串
// 题解
// 中等

// # 3哈希表字符串滑动窗口
// 串联所有单词的子串
// 题解
// 困难

// # 30哈希表字符串滑动窗口
// 最小覆盖子串
// 题解
// 困难

// # 76哈希表字符串滑动窗口
// 矩阵
// 有效的数独
// 题解
// 中等

// # 36数组哈希表矩阵
// 螺旋矩阵
// 题解
// 中等

// # 54数组矩阵模拟
// 旋转图像
// 题解
// 中等

// # 48数组数学矩阵
// 矩阵置零
// 题解
// 中等

// # 73数组哈希表矩阵
// 生命游戏
// 题解
// 中等

// # 289数组矩阵模拟
// 哈希表
// 赎金信
// 题解
// 简单

// # 383哈希表字符串计数
// 同构字符串
// 题解
// 简单

// # 205哈希表字符串
// 单词规律
// 题解
// 简单

// # 290哈希表字符串
// 有效的字母异位词
// 题解
// 简单

// # 242哈希表字符串排序
// 字母异位词分组
// 题解
// 中等

// # 49数组哈希表字符串1+
// 两数之和
// 题解
// 简单

// # 1数组哈希表
// 快乐数
// 题解
// 简单

// # 202哈希表数学双指针
// 存在重复元素 II
// 题解
// 简单

// # 219数组哈希表滑动窗口
// 最长连续序列
// 题解
// 中等

// # 128并查集数组哈希表
// 区间
// 汇总区间
// 题解
// 简单

// # 228数组
// 合并区间
// 题解
// 中等

// # 56数组排序
// 插入区间
// 题解
// 中等

// # 57数组
// 用最少数量的箭引爆气球
// 题解
// 中等

// # 452贪心数组排序
// 栈
// 有效的括号
// 题解
// 简单

// # 20栈字符串
// 简化路径
// 题解
// 中等

// # 71栈字符串
// 最小栈
// 题解
// 中等

// # 155栈设计
// 逆波兰表达式求值
// 题解
// 中等

// # 150栈数组数学
// 基本计算器
// 题解
// 困难

// # 224栈递归数学1+
// 链表
// 环形链表
// 题解
// 简单

// # 141哈希表链表双指针
// 两数相加
// 题解
// 中等

// # 2递归链表数学
// 合并两个有序链表
// 题解
// 简单

// # 21递归链表
// 随机链表的复制
// 题解
// 中等

// # 138哈希表链表
// 反转链表 II
// 题解
// 中等

// # 92链表
// K 个一组翻转链表
// 题解
// 困难

// # 25递归链表
// 删除链表的倒数第 N 个结点
// 题解
// 中等

// # 19链表双指针
// 删除排序链表中的重复元素 II
// 题解
// 中等

// # 82链表双指针
// 旋转链表
// 题解
// 中等

// # 61链表双指针
// 分隔链表
// 题解
// 中等

// # 86链表双指针
// LRU 缓存
// 题解
// 中等

// # 146设计哈希表链表1+
// 二叉树
// 二叉树的最大深度
// 简单

// # 104树深度优先搜索广度优先搜索1+
// 相同的树
// 简单

// # 100树深度优先搜索广度优先搜索1+
// 翻转二叉树
// 简单

// # 226树深度优先搜索广度优先搜索1+
// 对称二叉树
// 简单

// # 101树深度优先搜索广度优先搜索1+
// 从前序与中序遍历序列构造二叉树
// 中等

// # 105树数组哈希表2+
// 从中序与后序遍历序列构造二叉树
// 中等

// # 106树数组哈希表2+
// 填充每个节点的下一个右侧节点指针 II
// 中等

// # 117树深度优先搜索广度优先搜索2+
// 二叉树展开为链表
// 中等

// # 114栈树深度优先搜索2+
// 路径总和
// 简单

// # 112树深度优先搜索广度优先搜索1+
// 求根节点到叶节点数字之和
// 中等

// # 129树深度优先搜索二叉树
// 二叉树中的最大路径和
// 困难

// # 124树深度优先搜索动态规划1+
// 二叉搜索树迭代器
// 中等

// # 173栈树设计3+
// 完全二叉树的节点个数
// 简单

// # 222位运算树二分查找1+
// 二叉树的最近公共祖先
// 中等

// # 236树深度优先搜索二叉树
// 二叉树层次遍历
// 二叉树的右视图
// 中等

// # 199树深度优先搜索广度优先搜索1+
// 二叉树的层平均值
// 简单

// # 637树深度优先搜索广度优先搜索1+
// 二叉树的层序遍历
// 中等

// # 102树广度优先搜索二叉树
// 二叉树的锯齿形层序遍历
// 中等

// # 103树广度优先搜索二叉树
// 二叉搜索树
// 二叉搜索树的最小绝对差
// 简单

// # 530树深度优先搜索广度优先搜索2+
// 二叉搜索树中第 K 小的元素
// 中等

// # 230树深度优先搜索二叉搜索树1+
// 验证二叉搜索树
// 中等

// # 98树深度优先搜索二叉搜索树1+
// 图
// 岛屿数量
// 中等

// # 200深度优先搜索广度优先搜索并查集2+
// 被围绕的区域
// 中等

// # 130深度优先搜索广度优先搜索并查集2+
// 克隆图
// 中等

// # 133深度优先搜索广度优先搜索图1+
// 除法求值
// 中等

// # 399深度优先搜索广度优先搜索并查集4+
// 课程表
// 中等

// # 207深度优先搜索广度优先搜索图1+
// 课程表 II
// 中等

// # 210深度优先搜索广度优先搜索图1+
// 图的广度优先搜索
// 蛇梯棋
// 中等

// # 909广度优先搜索数组矩阵
// 最小基因变化
// 中等

// # 433广度优先搜索哈希表字符串
// 单词接龙
// 困难

// # 127广度优先搜索哈希表字符串
// 字典树
// 实现 Trie (前缀树)
// 中等

// # 208设计字典树哈希表1+
// 添加与搜索单词 - 数据结构设计
// 中等

// # 211深度优先搜索设计字典树1+
// 单词搜索 II
// 困难

// # 212字典树数组字符串2+
// 回溯
// 电话号码的字母组合
// 中等

// # 17哈希表字符串回溯
// 组合
// 中等

// # 77回溯
// 全排列
// 中等

// # 46数组回溯
// 组合总和
// 中等

// # 39数组回溯
// N 皇后 II
// 困难

// # 52回溯
// 括号生成
// 中等

// # 22字符串动态规划回溯
// 单词搜索
// 中等

// # 79深度优先搜索数组字符串2+
// 分治
// 将有序数组转换为二叉搜索树
// 简单

// # 108树二叉搜索树数组2+
// 排序链表
// 中等

// # 148链表双指针分治2+
// 建立四叉树
// 中等

// # 427树数组分治1+
// 合并 K 个升序链表
// 困难

// # 23链表分治堆（优先队列）1+
// Kadane 算法
// 最大子数组和
// 中等

// # 53数组分治动态规划
// 环形子数组的最大和
// 中等

// # 918队列数组分治2+
// 二分查找
// 搜索插入位置
// 简单

// # 35数组二分查找
// 搜索二维矩阵
// 中等

// # 74数组二分查找矩阵
// 寻找峰值
// 中等

// # 162数组二分查找
// 搜索旋转排序数组
// 中等

// # 33数组二分查找
// 在排序数组中查找元素的第一个和最后一个位置
// 中等

// # 34数组二分查找
// 寻找旋转排序数组中的最小值
// 中等

// # 153数组二分查找
// 寻找两个正序数组的中位数
// 困难

// # 4数组二分查找分治
// 堆
// 数组中的第K个最大元素
// 中等

// # 215数组分治快速选择2+
// IPO
// 困难

// # 502贪心数组排序1+
// 查找和最小的 K 对数字
// 中等

// # 373数组堆（优先队列）
// 数据流的中位数
// 困难

// # 295设计双指针数据流2+
// 位运算
// 二进制求和
// 简单

// # 67位运算数学字符串1+
// 颠倒二进制位
// 简单

// # 190位运算分治
// 位1的个数
// 简单

// # 191位运算分治
// 只出现一次的数字
// 简单

// # 136位运算数组
// 只出现一次的数字 II
// 中等

// # 137位运算数组
// 数字范围按位与
// 中等

// # 201位运算
// 数学
// 回文数
// 简单

// # 9数学
// 加一
// 简单

// # 66数组数学
// 阶乘后的零
// 中等

// # 172数学
// x 的平方根
// 简单

// # 69数学二分查找
// Pow(x, n)
// 中等

// # 50递归数学
// 直线上最多的点数
// 困难

// # 149几何数组哈希表1+
// 一维动态规划
// 爬楼梯
// 简单

// # 70记忆化搜索数学动态规划
// 打家劫舍
// 中等

// # 198数组动态规划
// 单词拆分
// 中等

// # 139字典树记忆化搜索数组3+
// 零钱兑换
// 中等

// # 322广度优先搜索数组动态规划
// 最长递增子序列
// 中等

// # 300数组二分查找动态规划
// 多维动态规划
// 三角形最小路径和
// 中等

// # 120数组动态规划
// 最小路径和
// 中等

// # 64数组动态规划矩阵
// 不同路径 II
// 中等

// # 63数组动态规划矩阵
// 最长回文子串
// 中等

// # 5双指针字符串动态规划
// 交错字符串
// 中等

// # 97字符串动态规划
// 编辑距离
// 中等

// # 72字符串动态规划
// 买卖股票的最佳时机 III
// 困难

// # 123数组动态规划
// 买卖股票的最佳时机 IV
// 困难

// # 188数组动态规划
// 最大正方形
// 中等

// # 221数组动态规划矩阵
//0 1 2 3 4 5 6 7
//a b a c a b a b

//a b a b a c a
//0 0 1 2 3 0 1

//a b c b a b c a
//0 0 0 0 1 2 3 1
List<int> buildNextArray2(String pattern) {
  final next = List<int>.filled(pattern.length, 0);

  // i: 当前前缀指针，同时表示最长公共前缀后缀长度
  for (int i = 0, j = 1; j < pattern.length; j++) {
    // 当字符不匹配时回退指针
    while (i > 0 && pattern[j] != pattern[i]) {
      i = next[i - 1];
    }

    // 当字符匹配时更新最长公共长度
    if (pattern[j] == pattern[i]) {
      i++;
    }

    // 设置当前位置的next值
    next[j] = i;
  }
  return next;
}

// KMP主搜索函数
int kmpSearch(String text, String pattern) {
  if (pattern.isEmpty) return 0; // 空模式直接返回0
  if (text.isEmpty) return -1; // 空文本无匹配

  final next = buildNextArray2(pattern); // 构建Next数组

  int j = 0; // 模式串索引
  for (int i = 0; i < text.length; i++) {
    // i: 文本串索引
    // 当匹配失败且j>0时，回退到Next数组指示的位置
    while (j > 0 && text[i] != pattern[j]) {
      j = next[j - 1];
    }
    // 当前字符匹配成功，移动模式串指针
    if (text[i] == pattern[j]) {
      j++;
    }
    // 完整匹配成功
    if (j == pattern.length) {
      return i - j + 1; // 返回起始位置
    }
  }

  return -1; // 未找到匹配
}
// 下标  0 1 2 3 4 5 6 7 8
// 文本  A B A B A B A B C

// 模式:         A B A B C

// 下标: 0 0 1 2 0

int strStr(String haystack, String needle) {
  List<int> next = List.filled(needle.length, 0);
  void buildNext(String needle) {
    for (int i = 0, j = 1; j < needle.length; j++) {
      while (i > 0 && needle[i] != needle[j]) {
        i = next[i - 1];
      }
      if (needle[i] == needle[j]) i++;
      next[j] = i;
    }
  }

  buildNext(needle); // build the next array for the needle
  for (int i = 0, j = 0; i < haystack.length; i++) {
    while (j > 0 && haystack[i] != needle[j]) {
      j = next[
          j - 1]; // if mismatch, move j to the next position in the next array
    }
    if (haystack[i] == needle[j]) j++;
    if (j == needle.length) return i - j + 1;
  }

  return -1;
}

//[["1","0","1","0","0"],
// ["1","0","1","1","1"],
// ["1","1","1","1","1"],
// ["1","0","0","1","0"]]

/**
 347. 前 K 个高频元素
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
输入：nums = [1,1,1,2,2,3], k = 2

输出：[1,2]
 */
List<int> topKFrequent(List<int> nums, int k) {
  List<int> result = [];
  HeapPriorityQueue<Map<int, int>> pq = HeapPriorityQueue((a, b) {
    return b.values.first - a.values.first;
  });
  Map<int, int> countMap = {};
  for (int num in nums) {
    countMap[num] = (countMap[num] ?? 0) + 1;
  }
  countMap.forEach((key, value) {
    pq.add({key: value});
  });
  for (int i = 0; i < k; i++) {
    result.add(pq.removeFirst().keys.first);
  }
  return result;
}
/**
 455. 分发饼干
假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；
并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是满足尽可能多的孩子，并输出这个最大数值。
示例 1:

输入: g = [1,2,3], s = [1,1]
输出: 1
解释: 
 */

int findContentChildren(List<int> g, List<int> s) {
  //思路：局部最优，大饼干先分配给大胃口避免浪费
  int res = 0;
  g.sort();
  s.sort();
  for (int i = g.length - 1, j = s.length - 1; i >= 0 && j >= 0; i--) {
    if (g[i] <= s[j]) {
      j--;
      res++;
    }
  }
  return res;
}

/**
 376. 摆动序列
力扣题目链接

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。

例如， [1,7,4,9,2,5] 是一个摆动序列，因为差值 (6,-3,5,-7,3)  是正负交替出现的。相反, [1,4,7,2,5]  和  [1,7,4,5,5] 不是摆动序列，
第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。

给定一个整数序列，返回作为摆动序列的最长子序列的长度。 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。

示例 1:

输入: [1,7,4,9,2,5]
输出: 6
解释: 整个序列均为摆动序列。
 */
int wiggleMaxLength(List<int> nums) {
  if (nums.length < 2) return nums.length;
  int res = 1;
  int lastDiff = 0;
  int currentDiff = 0;
  for (int i = 1; i < nums.length; i++) {
    currentDiff = nums[i] - nums[i - 1];
    if (lastDiff >= 0 && currentDiff < 0 || lastDiff <= 0 && currentDiff > 0) {
      res++;
      lastDiff = currentDiff;
    }
  }
  return res;
}

/**
 763.划分字母区间
字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

示例：

输入：S = "ababcbacadefegdehijhklij"
输出：[9,7,8] 解释： 划分结果为 "ababcbaca", "defegde", "hijhklij"。 
每个字母最多出现在一个片段中。 像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
提示：

S的长度在[1, 500]之间。
S只包含小写字母 'a' 到 'z' 。
 */
List<int> partitionLabels(String s) {
  List<int> myMap = List.filled(26, 0);
  List<int> res = [];
  for (int i = 0; i < s.length; i++) {
    myMap[s[i].codeUnitAt(0) - 'a'.codeUnitAt(0)] = i;
  }
  int end = 0;
  int last = -1;
  for (int i = 0; i < s.length; i++) {
    end = max(end, myMap[s[i].codeUnitAt(0) - 'a'.codeUnitAt(0)]);
    if (end == i) {
      res.add(end - last);
      last = end;
    }
  }
  return res;
}

/**
 509. 斐波那契数
斐波那契数 （通常用 F(n) 表示）形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：

F(0) = 0，F(1) = 1
F(n) = F(n - 1) + F(n - 2)，其中 n > 1
给定 n ，请计算 F(n) 。
输入：n = 2
输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1
 */
int fib(int n) {
  // if (n == 0) return 0;
  // if (n == 1) return 1;
  // return fib(n - 1) + fib(n - 2);
  List<int> dp = List.filled(n + 1, 0);
  dp[0] = 0;
  if (n == 0) return dp[0];
  dp[1] = 1;
  for (int i = 2; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2];
  }
  return dp[n];
}

/**
 416. 分割等和子集
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
示例 2：

输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
 */
bool canPartition(List<int> nums) {
  int sum = nums.reduce((a, b) => a + b);
  if (sum % 2 != 0) return false;
  int target = sum ~/ 2;
  List<int> dp = List.filled(target + 1, 0);
  for (int i = 0; i < nums.length; i++) {
    for (int j = target; j >= nums[i]; j--) {
      // 01背包, 往target大小的背包中放入最大价值的物品，物品的价值就是nums[i]
      dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]);
    }
  }
  // 结果只有两种，要么能装满背包，要么不能，不可能价值比背包的容量还大
  return dp[target] == target;
}

/**
 1049.最后一块石头的重量II
有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；

如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。

最后，最多只会剩下一块石头。返回此石头最小的可能重量。如果没有石头剩下，就返回 0。

示例：

输入：[2,7,4,1,8,1]
输出：1
解释：

组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。
 */

int lastStoneWeightII(List<int> stones) {
  //思路：将石头分为两堆，重量差最小
  int sum = stones.reduce((a, b) => a + b);
  int target = sum ~/ 2;
  List<int> dp = List.filled(target + 1, 0);

  for (int i = 0; i < stones.length; i++) {
    for (int j = target; j >= stones[i]; j--) {
      dp[j] = max(dp[j], dp[j - stones[i]] + stones[i]);
    }
  }
  // 结果只有两种，要么能装满背包，要么不能，不可能价值比背包的容量还大
  return sum - 2 * dp[target];
}

/**
 518. 零钱兑换 II
给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。
输入：amount = 5, coins = [1, 2, 5]
输出：4
解释：有四种方式可以凑成总金额：
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
 */
int change(int target, List<int> coins) {
  // List<List<int>> dp = List.generate(coins.length, List.filled(amount + 1, 0));
//   List<List<int>> dp =
//       List.generate(coins.length, (index) => List.filled(amount + 1, 0));

// // 第一列初始化
//   for (int i = 0; i < coins.length; i++) {
//     dp[i][0] = 1; //金额为0时，只有一种方式
//   }
//   //第一行初始化
//   for (int j = 1; j <= amount; j++) {
//     if (j % coins[0] == 0) dp[0][j] = 1;
//   }
//   for (int i = 1; i < coins.length; i++) {
//     for (int j = 1; j <= amount; j++) {
//       if (j < coins[i])
//         dp[i][j] = dp[i - 1][j];
//       else
//         dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i]];
//     }
//   }
//   return dp[coins.length - 1][amount];

  List<int> dp = List.filled(target + 1, 0);
  // 金额为0时，只有一种方式组成
  dp[0] = 1;
  for (int i = 0; i < coins.length; i++) {
    // int j = coins[i]是必要的，因为j< coins[i]时，容量放不下物品，而且dp[j - coins[i]]会越界
    for (int j = coins[i]; j <= target; j++) {
      //i=0 1 1 1 1 1
      //i=1 1 1 2 2 3
      // dp[j] 表示不放coins[i]时的方案数，也是i-1时的方案数
      // dp[j - coins[i]] 表示放coins[i]时的方案数
      dp[j] = dp[j] + dp[j - coins[i]];
    }
  }
  return dp[target];
}

/**
 322. 零钱兑换
力扣题目链接

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3
解释：11 = 5 + 5 + 1

提示：

1 <= coins.length <= 12
1 <= coins[i] <= 2^31 - 1
0 <= amount <= 10^4
 */

int coinChange(List<int> coins, int amount) {
  List<int> dp = List.filled(amount + 1, amount + 1);
  // 金额为0时，只需0个硬币就能组成金额为0
  dp[0] = 0;
  for (int i = 0; i < coins.length; i++) {
    for (int j = coins[i]; j <= amount; j++) {
      dp[j] = min(dp[j], dp[j - coins[i]] + 1);
    }
  }
  return dp[amount] > amount ? -1 : dp[amount];
}

/**
 279.完全平方数
给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。
输入：n = 12
输出：3
解释：12 = 4 + 4 + 4

1 <= n <= 10^4
 */
int numSquares(int n) {
  List<int> dp = List.filled(n + 1, n + 1);
  List<int> squares = [];
  for (int i = 1; i <= n; i++) {
    if (i * i > n) break;
    squares.add(i * i);
  }
  dp[0] = 0;
  for (int i = 0; i < squares.length; i++) {
    for (int j = squares[i]; j <= n; j++) {
      dp[j] = min(dp[j], dp[j - squares[i]] + 1);
    }
  }
  return dp[n] == n + 1 ? -1 : dp[n];
}

/*
377. 组合总和 Ⅳ
给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。
*/

/*
    0 1 2 3 4 容量
0 1 1 1 1 
1 2 1 1 
2 3 1 1
物
品

*/
int combinationSum4(List<int> nums, int target) {
  // List<List<int>> dp =
  //     List.generate(nums.length + 1, (index) => List.filled(target + 1, 0));

  // for (int i = 0; i < nums.length; i++) {
  //   dp[i][0] = 1;
  // }
  // for (int j = 1; j <= target; j++) {
  //   for (int i = 1; i <= nums.length; i++) {
  //     if (j >= nums[i])
  //       //
  //       // 不放nums[i]
  //       // i = 0 时，dp[-1][j]恰好为0，所以没有特殊处理
  //       dp[i][j] = dp[i][j - 1] +
  //           //放nums[i]。取容量为j - nums[i]时的所有排列，举例：容量为102时，如果放入nums[i]=2，那么取容量为100时的所有排列，因为是完全背包，
  //所以容量为100的排列是所有元素1 2 3的排列。
  //           // 容量为100的排列是dp[3][100]
  //           dp[nums.length - 1][j - nums[i]];
  //     else
  //       dp[i][j] = dp[i - 1][j];
  //   }
  // }
  // print(dp);
  // return dp[nums.length][target];
  List<int> dp = List.filled(target + 1, 0);
  for (int j = 1; j <= target; j++) {
    for (int i = 0; i < nums.length; i++) {
      //这个是所有背包问题必须的，j< nums[i]时，容量放不下物品，而且dp[j - nums[i]]会越界
      if (j >= nums[i]) {
        // dp[j]表示容量为j时的所有排列,
        // 后面的dp[j] 表示不放入nums[i]时，i-1时的容量为j的所有排列
        // dp[j - nums[i]]表示放入nums[i]时，容量为j
        dp[j] = dp[j] + dp[j - nums[i]];
      }
    }
  }

  return dp[target];
}

/**
 198.打家劫舍
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。   偷窃到的最高金额 = 1 + 3 = 4 。
 */

int rob(List<int> nums) {
  if (nums.length == 1) return nums[0];
  List<int> dp = List.filled(nums.length, 0);
  dp[0] = nums[0];
  dp[1] = max(nums[0], nums[1]);
  for (int i = 2; i < nums.length; i++) {
    dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
  }
  return dp[nums.length - 1];
}

/**
 *213.打家劫舍II
你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。
同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。
输入：nums = [2,3,2]
输出：3
 */
int rob2(List<int> nums) {
  if (nums.length == 1) return nums[0];
  int withNoFirst = rob(nums.sublist(1));
  int withNoLast = rob(nums.sublist(0, nums.length - 1));
  return max(withNoFirst, withNoLast);
}
/**
 337.打家劫舍 III
这个地方的所有房屋的排列类似于一棵二叉树。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
 */

int rob3(TreeNode? root) {
  //0不偷，1偷
  List<int> robAction(TreeNode? node) {
    if (node == null) return [0, 0];
    List<int> left = robAction(node.left);
    List<int> right = robAction(node.right);
    List<int> res = List.filled(2, 0);
    // 不偷：Max(左孩子不偷，左孩子偷) + Max(右孩子不偷，右孩子偷)
    res[0] = max(left[0], left[1]) + max(right[0], right[1]);
    res[1] = node.val + left[0] + right[0];
    return res;
  }

  List<int> res = robAction(root);
  return max(res[0], res[1]);
}
/**
 300.最长递增子序列
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
输入：nums = [10,9,2,5,4,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
// 不能用单调递增，反例[0,3,1,-1，6,2,2,7]
 */

int lengthOfLIS(List<int> nums) {
  List<int> dp = List.filled(nums.length, 1);
  int res = 1;
  for (int i = 1; i < nums.length; i++) {
    for (int j = 0; j < i; j++) {
      if (nums[i] > nums[j]) {
        dp[i] = max(dp[i], dp[j] + 1);
      }
    }
    res = max(res, dp[i]);
  }
  return res;
}

/**
 674. 最长连续递增序列
给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。
连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，
那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。
输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。
 */
int findLengthOfLCIS(List<int> nums) {
  List<int> dp = List.filled(nums.length, 1);
  int res = 1;
  for (int i = 1; i < nums.length; i++) {
    if (nums[i] > nums[i - 1]) {
      dp[i] = dp[i - 1] + 1;
    }
    res = max(res, dp[i]);
  }
  return res;
}

/**
 718. 最长重复子数组
给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：长度最长的公共子数组是 [3, 2, 1] 。
提示：

1 <= len(A), len(B) <= 1000
0 <= A[i], B[i] < 100
 */

int findLength(List<int> nums1, List<int> nums2) {
  List<List<int>> dp = List.generate(
      nums1.length + 1, (index) => List.filled(nums2.length + 1, 0));
  int res = 0;
  for (int i = 1; i <= nums1.length; i++) {
    for (int j = 1; j <= nums2.length; j++) {
      if (nums1[i - 1] == nums2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
        res = max(res, dp[i][j]);
      }
    }
  }
  return res;
}

/**
 1143.最长公共子序列
力扣题目链接

给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

若这两个字符串没有公共子序列，则返回 0。

示例 1:

输入：text1 = "abcde", text2 = "aces"
输出：3
解释：最长公共子序列是 "ace"，它的长度为 3。
  0 a c e s
0 0 0 0 0 0
a 0 1 1 1 1
b 0 1 1 1 1
c 0 1 2 2 2
d 0 1 2 2 2
e 0 1 2 3 3
 */

int longestCommonSubsequence(String text1, String text2) {
  List<List<int>> dp = List.generate(
      text1.length + 1, (index) => List.filled(text2.length + 1, 0));
  for (int i = 1; i <= text1.length; i++) {
    for (int j = 1; j <= text2.length; j++) {
      if (text1[i - 1] == text2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  return dp[text1.length][text2.length];
}

/**
 392.判断子序列
给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
输入：s = "abc", t = "ahbgdc"
输出：true

输入：s = "axc", t = "ahbgdc"
输出：false
提示：
0 <= s.length <= 100
0 <= t.length <= 10^4
 */
bool isSubsequence(String s, String t) {
  // int i = 0, j = 0;
  // while (i < s.length && j < t.length) {
  //   if (s[i] == t[j]) i++;
  //   j++;
  // }
  // return i == s.length;
  List<List<int>> dp =
      List.generate(s.length + 1, (index) => List.filled(t.length + 1, 0));
  for (int i = 1; i <= s.length; i++) {
    for (int j = 1; j <= t.length; j++) {
      if (s[i - 1] == t[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = dp[i][j - 1];
      }
    }
  }
  return dp[s.length][t.length] == s.length;
}

/**
 * 115. 不同的子序列
给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数。
测试用例保证结果在 32 位有符号整数范围内。
输入：s = "rabbbit", t = "rabbit"
输出：3
如下所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
rabbbit
rabbbit
rabbbit

输入：s = "babgbag", t = "bag"
输出：5

如下所示, 有 5 种可以从 s 中得到 "bag" 的方案。 
babgbag
babgbag
babgbag
babgbag
babgbag
 */
int numDistinct(String s, String t) {
  List<List<int>> dp =
      List.generate(s.length + 1, (index) => List.filled(t.length + 1, 0));
  for (int i = 0; i <= s.length; i++) dp[i][0] = 1;
  for (int i = 1; i <= s.length; i++) {
    for (int j = 1; j <= t.length; j++) {
      if (s[i - 1] == t[j - 1])
        dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
      else
        dp[i][j] = dp[i - 1][j];
    }
  }
  return dp[s.length][t.length];
}

/**
 583. 两个字符串的删除操作
给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。
输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
 */

int minDistance(String word1, String word2) {
  List<List<int>> dp = List.generate(
      word1.length + 1, (index) => List.filled(word2.length + 1, 0));
  for (int i = 1; i <= word1.length; i++) dp[i][0] = i;
  for (int j = 1; j <= word2.length; j++) dp[0][j] = j;
  for (int i = 1; i <= word1.length; i++) {
    for (int j = 1; j <= word2.length; j++) {
      if (word1[i - 1] == word2[j - 1])
        dp[i][j] = dp[i - 1][j - 1];
      else
        dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
    }
  }
  return dp[word1.length][word2.length];
}
/**
 72. 编辑距离
给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
你可以对一个单词进行如下三种操作：
插入一个字符
删除一个字符
替换一个字符

输入：word1 = "horse", word2 = "ros"
输出：3
解释： horse -> rorse (将 'h' 替换为 'r') rorse -> rose (删除 'r') rose -> ros (删除 'e')

0 <= word1.length, word2.length <= 500
word1 和 word2 由小写英文字母组成
 */

int minDistance1(String word1, String word2) {
  List<List<int>> dp = List.generate(
      word1.length + 1, (index) => List.filled(word2.length + 1, 0));
  for (int i = 0; i <= word1.length; i++) dp[i][0] = i;
  for (int j = 0; j <= word2.length; j++) dp[0][j] = j;
  for (int i = 1; i <= word1.length; i++) {
    for (int j = 1; j <= word2.length; j++) {
      if (word1[i - 1] == word2[j - 1])
        dp[i][j] = dp[i - 1][j - 1];
      else
        dp[i][j] = min(
            // 换word1的字符
            dp[i - 1][j - 1] + 1,
            // 删除word1的字符，效果等于往word2插入一个字符
            min(
                dp[i - 1][j] + 1,
                // 删除word2的字符，效果等于往word1插入一个字符
                dp[i][j - 1] + 1));
    }
  }
  return dp[word1.length][word2.length];
}

/**
 647. 回文子串
给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
输入："abc"
输出：3
解释：三个回文子串: "a", "b", "c"

输入："aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
提示：输入的字符串长度不会超过 1000 。
 */

int countSubstrings(String s) {
  int res = 0;
  List<List<bool>> dp =
      List.generate(s.length, (index) => List.filled(s.length, false));
  for (int i = s.length - 1; i >= 0; i--) {
    for (int j = i; j < s.length; j++) {
      if (s[i] == s[j]) {
        if (j - i <= 1) {
          res++;
          dp[i][j] = true;
        } else if (dp[i + 1][j - 1]) {
          res++;
          dp[i][j] = true;
        }
      }
    }
  }
  return res;
}

/**
 516.最长回文子序列
给定一个字符串 s ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 s 的最大长度为 1000 。
示例 1: 输入: "bbbab" 输出: 4 一个可能的最长回文子序列为 "bbbb"。
示例 2: 输入:"cbbd" 输出: 2 一个可能的最长回文子序列为 "bb"。
    0 1 2 3 
    a b b b 
0 a 1 1 1 1
1 b 0 1 2 3
2 b 0 0 1 2
3 b 0 0 0 0
 */

int longestPalindromeSubseq(String s) {
  List<List<int>> dp =
      List.generate(s.length, (index) => List.filled(s.length, 0));
  for (int i = s.length - 1; i >= 0; i--) {
    dp[i][i] = 1;
    for (int j = i + 1; j < s.length; j++) {
      if (s[i] == s[j]) {
        dp[i][j] = dp[i + 1][j - 1] + 2;
      } else {
        dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
      }
    }
  }
  return dp[0][s.length - 1];
}

/**
 739. 每日温度
请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。
例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
 */

List<int> dailyTemperatures(List<int> temperatures) {
  if (temperatures.isEmpty) return [];
  List<int> res = List.filled(temperatures.length, 0);
  List<int> stack = [];
  stack.add(0);
  for (int i = 1; i < temperatures.length; i++) {
    if (temperatures[i] <= temperatures[stack.last])
      stack.add(i);
    else {
      while (stack.isNotEmpty && temperatures[i] > temperatures[stack.last]) {
        res[stack.last] = i - stack.last;
        stack.removeLast();
      }
      stack.add(i);
    }
  }
  return res;
}

/**
 496.下一个更大元素 I
力扣题目链接

给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。

请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。

nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。

示例 1:

输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [-1,3,-1]
解释:
对于 num1 中的数字 4 ，你无法在第二个数组中找到下一个更大的数字，因此输出 -1 。
对于 num1 中的数字 1 ，第二个数组中数字1右边的下一个较大数字是 3 。
对于 num1 中的数字 2 ，第二个数组中没有下一个更大的数字，因此输出 -1 。

 */
List<int> nextGreaterElement(List<int> nums1, List<int> nums2) {
  Map<int, int> myMap = {};
  List<int> stack = [];
  stack.add(0);
  myMap[nums2[0]] = 0;
  for (int i = 1; i < nums2.length; i++) {
    myMap[nums2[i]] = i;
    if (nums2[i] <= nums2[stack.last]) {
      stack.add(i);
    } else {
      while (stack.isNotEmpty && nums2[i] > nums2[stack.last]) {
        nums2[stack.last] = nums2[i];
        stack.removeLast();
      }
      stack.add(i);
    }
  }
  while (stack.isNotEmpty) {
    nums2[stack.last] = -1;
    stack.removeLast();
  }
  for (int i = 0; i < nums1.length; i++) {
    nums1[i] = nums2[myMap[nums1[i]]!];
  }
  return nums1;
}

/**
 503.下一个更大元素II
给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，
这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。

示例 1:

输入: [1,2,1]
输出: [2,-1,2]
解释: 第一个 1 的下一个更大的数是 2；数字 2 找不到下一个更大的数；第二个 1 的下一个最大的数需要循环搜索，结果也是 2。
提示:

1 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9

[0 5,4,3,2,1 0 5 4 3 2 1]
[5]
[5 4 3 2 1 0]
[-1,5,5,5,5]
 */
List<int> nextGreaterElements(List<int> nums) {
  List<int> res = List.filled(nums.length, -1);
  List<int> stack = [];
  stack.add(0);
  for (int i = 1; i < nums.length * 2; i++) {
    if (nums[i % nums.length] > nums[stack.last]) {
      while (stack.isNotEmpty && nums[i % nums.length] > nums[stack.last]) {
        res[stack.last] = nums[i % nums.length];
        stack.removeLast();
      }
    }
    stack.add(i % nums.length);
  }
  return res;
}

/*
关于二叉树，你该了解这些！
二叉树：二叉树的递归遍历          inorder、 preorder、 postorder
二叉树：二叉树的迭代遍历          inorder1、 preorder1、postorder1
二叉树：二叉树的统一迭代法
二叉树：二叉树的层序遍历          levelOrder
二叉树：226.翻转二叉树           invertTree  
本周小结！（二叉树）
二叉树：101.对称二叉树           isSymmetric
二叉树：104.二叉树的最大深度      maxDepth
二叉树：111.二叉树的最小深度      minDepth
二叉树：222.完全二叉树的节点个数   countNodes
二叉树：110.平衡二叉树         isBalanced
二叉树：257.二叉树的所有路径     binaryTreePaths
本周总结！（二叉树）
二叉树：404.左叶子之和        sumOfLeftLeaves
二叉树：513.找树左下角的值     findBottomLeftValue
二叉树：112.路径总和          hasPathSum
二叉树：106.构造二叉树       buildTree
二叉树：654.最大二叉树     constructMaximumBinaryTree //
本周小结！（二叉树）
二叉树：617.合并两个二叉树 mergeTrees//
二叉树：700.二叉搜索树登场，找到树中对应的值 searchBST
二叉树：98.验证二叉搜索树     isValidBST
二叉树：530.搜索树的最小绝对差 getMinimumDifference//
二叉树：501.二叉搜索树中的众数 findMode//
二叉树：236.公共祖先问题 lowestCommonAncestor//
本周小结！（二叉树）
二叉树：235.搜索树的最近公共祖先 lowestCommonAncestor1//
二叉树：701.搜索树中的插入操作 insertIntoBST//
二叉树：450.搜索树中的删除操作 deleteNode//
二叉树：669.修剪二叉搜索树 trimBST//
二叉树：108.将有序数组转换为二叉搜索树 sortedArrayToBST
二叉树：538.把二叉搜索树转换为累加树 convertBST
二叉树：总结篇！

*/

/**
 关于回溯算法，你该了解这些！
回溯算法：77.组合            combine
回溯算法：216.组合总和III.    combinationSum3
回溯算法：17.电话号码的字母组合 letterCombinations
本周小结！（回溯算法系列一）
回溯算法：39.组合总和         combinationSum
回溯算法：40.组合总和II    combinationSum2
回溯算法：131.分割回文串   partition
回溯算法：93.复原IP地址 restoreIpAddresses
回溯算法：78.子集         subsets
本周小结！（回溯算法系列二）
回溯算法：90.子集II       subsetsWithDup
回溯算法：491.递增子序列 findSubsequences
回溯算法：46.全排列     permute
回溯算法：47.全排列II   permuteUnique
本周小结！（回溯算法系列三）
回溯算法去重问题的另一种写法
回溯算法：51.N皇后       solveNQueens
回溯算法：37.解数独      solveSudoku
回溯算法总结篇
 */
/*
关于贪心算法，你该了解这些！
贪心算法：455.分发饼干   findContentChildren
贪心算法：376.摆动序列   wiggleMaxLength
贪心算法：53.最大子序和.  maxSubArray
本周小结！（贪心算法系列一）
贪心算法：122.买卖股票的最佳时机II maxProfit
贪心算法：55.跳跃游戏   canJump
贪心算法：45.跳跃游戏II jump
贪心算法：1005.K次取反后最大化的数组和 largestSumAfterKNegations
本周小结！（贪心算法系列二）
贪心算法：134.加油站    canCompleteCircuit
贪心算法：135.分发糖果  candy
贪心算法：860.柠檬水找零 lemonadeChange
贪心算法：406.根据身高重建队列 reconstructQueue
本周小结！（贪心算法系列三）
贪心算法：435.无重叠区间    eraseOverlapIntervals
贪心算法：763.划分字母区间  partitionLabels
贪心算法：56.合并区间       merge
本周小结！（贪心算法系列四）
贪心算法：738.单调递增的数字 monotoneIncreasingDigits

*/
/**
 * 
 * // 动态规划最重要的三要素
1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
动态规划：509.斐波那契数.   fib
动态规划：70.爬楼梯         climbStairs
动态规划：746.使用最小花费爬楼梯 minCostClimbingStairs
本周小结！（动态规划系列一）
动态规划：62.不同路径     uniquePaths
动态规划：63.不同路径II  uniquePathsWithObstacles
动态规划：343.整数拆分 integerBreak
动态规划：96.不同的二叉搜索树  numTrees
本周小结！（动态规划系列二）


动态规划：01背包理论基础（二维dp数组）背包某容量最多能装下多少价值的物品
动态规划：01背包理论基础（一维dp数组）背包某容量最多能装下多少价值的物品
动态规划：416.分割等和子集          canPartition， 一半sum大小的背包能否装满，能装满则true
动态规划：1049.最后一块石头的重量II  lastStoneWeightII， 一半sum大小的背包能装下多重的石头，返回sum-2*dp[target]
本周小结！（动态规划系列三）
动态规划：494.目标和               findTargetSumWays  left  = target + right, right = sum - left, left = (sum + target) / 2, left为正数，right为负数
                                left 为背包容量，nums为物品重量，求有多少种组合能装满背包，不能重复使用物品，则一维数组倒叙便利
动态规划：完全背包理论基础（二维dp数组） 背包某容量最多能装下多少价值的物品，物品可以重复使用
动态规划：完全背包理论基础（一维dp数组）
动态规划：518.零钱兑换II           change ，背包容量amount，物品重量coins，求有多少种组合能装满背包，物品可以重复使用
本周小结！（动态规划系列四）
动态规划：377.组合总和Ⅳ           combinationSum4 ，背包容量target，物品重量nums，求有多少种排列能装满背包，物品可以重复使用， 排列，先循环target，再循环物品
动态规划：70.爬楼梯（完全背包版本）  climbStairs ，阶梯->背包容量n，跳跃选择->物品重量1,2..m，求有多少种排列能装满背包，物品可以重复使用，排列，先循环target，再循环物品
动态规划：322.零钱兑换             coinChange ，背包容量amount，物品coins，求 最少 多少物品能装满背包，物品可以重复使用
动态规划：279.完全平方数           numSquares ，背包容量n，物品1,4,9..m，求 最少 多少物品能装满背包，物品可以重复使用
本周小结！（动态规划系列五）
动态规划：139.单词拆分             wordBreak , s为背包容量，wordDict为物品，求能否装满背包，物品可以重复使用，而且是排列

打家劫舍系列：                    
动态规划：198.打家劫舍             rob    dp表示0-i房子所能偷的最大的金钱，    dp[i] = max(dp[i-1], dp[i-2]+nums[i]),
动态规划：213.打家劫舍II           rob2   dp表示0-i房子所能偷的最大的金钱，分两种情况考虑0-len-1， 1-len，    dp[i] = max(dp[i-1], dp[i-2]+nums[i]),
动态规划：337.打家劫舍III          rob3   dp表示以i为根节点的子树所能偷的最大的金钱，dp[i][0]表示不偷i节点，dp[i][1]表示偷i节点

股票系列：
动态规划：121.买卖股票的最佳时机    maxProfit， dp[i][0]表示第i天持有股票后手中剩余的最大金钱, dp[i][1]表示第i天不持有股票后手中剩余的最大金钱
动态规划：本周小结（系列六）
动态规划：122.买卖股票的最佳时机II  maxProfit2 ， dp[i][0]表示第i天持有股票后手中剩余的最大金钱, dp[i][1]表示第i天不持有股票后手中剩余的最大金钱
动态规划：123.买卖股票的最佳时机III maxProfit3 ， dp[i][0]表示第i天第一次持有股票后手中剩余的最大金钱, dp[i][1]表示第i天第一次不持有股票后手中剩余的最大金钱
                                             dp[i][2]表示第i天第二次持有股票后手中剩余的最大金钱, dp[i][3]表示第i天第二次不持有股票后手中剩余的最大金钱
动态规划：188.买卖股票的最佳时机IV  maxProfit4 ， k次交易，
动态规划：本周小结（系列七）
动态规划：714.买卖股票的最佳时机含手续费  maxProfit5 ， dp[i][0]表示第i天持有股票后手中剩余的最大金钱, dp[i][1]表示第i天不持有股票后手中剩余的最大金钱然后在减去手续费
动态规划：股票系列总结篇


子序列系列：
动态规划：300.最长递增子序列       lengthOfLIS dp[i]表示以nums[i]结尾的最长递增子序列的长度, if nums[i]>nums[j] dp[i] = max(dp[i], dp[j]+1) 
动态规划：674.最长连续递增序列     findLengthOfLCIS dp[i]表示以nums[i]结尾的最长 连续 递增子序列的长度, if nums[i]>nums[i-1] dp[i] = dp[i-1]+1
动态规划：718.最长重复子数组       findLength dp[i][j]表示以A[i-1]和B[j-1]结尾的最长重复子数组的长度, if A[i-1]==B[j-1] dp[i][j] = dp[i-1][j-1]+1
动态规划：1143.最长公共子序列     longestCommonSubsequence dp[i][j]表示以text1[i-1]和text2[j-1]结尾的最长公共子序列的长度, if text1[i-1]==text2[j-1] dp[i][j] = dp[i-1][j-1]+1 else dp[i][j] = max(dp[i-1][j], dp[i][j-1])

动态规划：53.最大子序和          maxSubArray dp[i]表示以nums[i]结尾的最大子数组和, dp[i] = max(dp[i-1]+nums[i], nums[i])
动态规划：392.判断子序列         isSubsequence dp[i][j]表示以text1[i-1]和text2[j-1]结尾的最长公共子序列的长度, if s[i-1]==t[j-1] dp[i][j] = dp[i-1][j-1]+1 else dp[i][j] = dp[i][j-1]
动态规划：115.不同的子序列        numDistinct dp[i][j]表示以s[i-1]和t[j-1]结尾的不同子序列的个数, if s[i-1]==t[j-1] dp[i][j] = dp[i-1][j-1]+dp[i-1][j] else dp[i][j] = dp[i-1][j]
动态规划：583.两个字符串的删除操作   minDistance dp[i][j]表示将word1[0-i-1]和word2[0-j-1]变成相同所需的最小步数, if word1[i-1]==word2[j-1] dp[i][j] = dp[i-1][j-1] else dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1)
动态规划：72.编辑距离              minDistance1 dp[i][j]表示将word1[0-i-1]和word2[0-j-1]变成相同所需的最小步数, if word1[i-1]==word2[j-1] dp[i][j] = dp[i-1][j-1] else dp[i][j] = min(dp[i-1][j-1]+1, min(dp[i-1][j]+1, dp[i][j-1]+1))
编辑距离总结篇
动态规划：647.回文子串            countSubstrings dp[i][j]表示s[i-j]是否为回文子串, if s[i]==s[j]  dp[i+1][j-1]) res++ dp[i][j] = true
动态规划：516.最长回文子序列  longestPalindromeSubseq dp[i][j]表示s[i-j]的最长回文子序列的长度
动态规划总结篇

单调栈
单调栈：739.每日温度            dailyTemperatures  单调递减
单调栈：496.下一个更大元素I      nextGreaterElement  单调递减
单调栈：503.下一个更大元素II     nextGreaterElements  单调递减
单调栈：42.接雨水               trap  单调递减，下一个更大元素使得栈顶元素变成凹糟
单调栈：84.柱状图中最大的矩形    largestRectangleArea  单调递增，下一个更小元素使得栈顶元素形成了一个矩形
 */

