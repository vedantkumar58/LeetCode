
This list contains Neetcode 150 problems :


242. Valid Anagram

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter

        count1=Counter(s)
        count2=Counter(t)
        if count1==count2:
            return(True)
        else:
            return(False)

        :
        
        
217. Contains Duplicate:

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
       
        nums=sorted(nums)
       
        for num in range(1,len(nums)):
            if nums[num]==nums[num-1]:
                
                return(True)
       
        return(False)

***************************************************
238. Product of Array Except Self:


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:

        n=len(nums)
        prefix=1
        suffix=1
        answer=[1] * n

        for i in range(n):
            answer[i]=prefix
            prefix *= nums[i]

        for i in range(n-1,-1,-1):
            answer[i] *= suffix
            suffix *= nums[i]
        
        return (answer)


      
Note : Small difference is that answer[i]=prefix  but answer[i] *=suffix
        
        
***************************************************


 11. Container with Most water :
 
 
 class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area=0
        n=len(height)
        left,right=0,n-1
        while left < right :
            if height[left] <= height[right] :
                area=height[left] * (right-left)
                max_area=max(max_area,area)
                left+=1

            else :
                area=height[right] * (right-left)
                max_area=max(max_area,area)
                right-=1
        return(max_area)


        
**********************


7. Reverse Integer

class Solution:
    def reverse(self, x: int) -> int:

        if x<0:

            x=int(str(abs(x))[::-1])
            if x >2**31-1:
                return(0)
            else:
                return(-x)
        else :
            x=int(str(x)[::-1])
            if x>2**31-1:
                return(0)
            else:
                return(x)

*********************************************


 
 TWO SUM :
 
 BRUTE FORCE :
 
 class Solution(object):
    def twoSum(self, nums, target):
        a=[]
        for i in range(len(nums)):
           
            for j in range(i+1,len(nums)):
                if (nums[i]+nums[j])==target:
                    
                    a=[i,j]
                    return a
        return []

Best method use dict/hash map

class Solution(object):
    def twoSum(self, nums, target):
        number_dict={}
        for i,num in enumerate(nums):
            compliment=target-num
            if compliment in number_dict:
                return(number_dict[compliment],i)
            number_dict[num]=i
        return[]

**************************
Two pointer approac(but best for questions where array is sorted not here)


class Solution(object):
    def twoSum(self, nums, target):
        nums = sorted((val,index) for index,val in enumerate(nums))
        left, right = 0, len(nums)-1
        while left <right :
            sum=nums[left][0]+nums[right][0]
            if sum==target:
                return(nums[left][1],nums[right][1])
            elif sum < target:
                left+=1
            else :
                right -=1
        return []


******************************************

Median of two sorted arrays:


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        arr=nums1+nums2
        arr=sorted(arr)

        n=len(arr)

        if n%2==1:
            median=arr[n//2]
            return(median)
        else :
            mid1=n//2
            mid2=n//2 -1
            return((arr[mid1]+arr[mid2])/2)
           
 153. Find Minimum in Rotated Sorted Array :       
        
class Solution:
    def findMin(self, nums: List[int]) -> int:

        l,r=0,len(nums)-1
        res=nums[0]

        while l <= r:
            if nums[l] <nums[r]:
                res=min(res,nums[l])
                break
            m= (l+r)//2
            res=min(res,nums[m])
            if nums[m] >= nums[l] :
                l=m+1
            else :
                r=m-1
        return (res)

 class Solution:
    def findMin(self, nums: List[int]) -> int:

        l,r=0,len(nums)-1
        

        while l < r:
            m= (l+r)//2
            if nums[m] > nums[r]:
                l=m+1
            else :
                r=m
        return(nums[l])

            
*****************************************************

20. Valid Parentheses

class Solution:
    def isValid(self, s: str) -> bool:
        
        stack=[]
        mapping = {')': '(', '}': '{', ']': '['}

        for char in s :
            if char in mapping :
                top_element=stack.pop() if stack else '#'
                    
                
                if mapping[char] != top_element:
                    return False
            else :
                stack.append(char)

        return not stack
        
        
**************************************************************************
                      

155. Min Stack

class MinStack:

    def __init__(self):
        self.stack = []
        

    def push(self, val: int) -> None:

        self.stack.append(val)
        

    def pop(self) -> None:
        
         return self.stack.pop()
        

    def top(self) -> int:

        return self.stack[-1]

 **********************************************
 739. Daily Temperatures
 
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:

        n= len(temperatures)
        stack=[]
        answer=[0] * n
        for i in range(n):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                index=stack.pop()
                answer[index]=i-index
            stack.append(i)

        return answer

 

       
*********************************************
150. Evaluate Reverse Polish Notation

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:

        stack =[]
       

        op = ['*', '+', '/', '-']

        for i in tokens :
            if i in op :
                if stack :
                    b=stack.pop()
                    a=stack.pop()
                    if i== '+':
                        stack.append(a+b)
                    elif i=='-':
                        stack.append(a-b)
                    elif i=='/':
                        stack.append(int(a/b))
                    elif i=='*':
                        stack.append(a*b)



            else :
                i=int(i)
                stack.append(i)
        return (stack[0])


***********************************************************************

CAR Fleet :


car=[[p,s] for p,s in zip(position,speed)]

        prev_time=0
        count=0

       
        for p, s in sorted(car, reverse=True):
    # Your logic here


            time_to_reach_destination=(target-p)/s
            if time_to_reach_destination > prev_time:
                prev_time=time_to_reach_destination
                count +=1
        return (count)

*************************************

121. Best Time to Buy and Sell Stock

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price=prices[0]

        max_profit=0

        for price in prices:
            min_price=min(min_price,price)
            profit = price -min_price
            max_profit=max(profit,max_profit)
        return(max_profit)
        
 
 
 TWO Pointer Approach :
 
 
 class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        

        max_profit=0

        left=0
        right=1

        while right < len(prices) :
            if prices[left] < prices[right]:
                profit = prices[right] - prices[left]
                max_profit=max(max_profit,profit)
                
            else :
                left=right
                
            right +=1

        return (max_profit)

******************************************************
3. Longest Substring Without Repeating Characters

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        n=len(s)
        char_set=set()
        left=0
        right=0
        count=0
        max_length=0

        for right in  range(n):
            while s[right] in char_set :
                char_set.remove(s[left])
                left +=1
            
            char_set.add(s[right])

            max_length = max(max_length, right - left + 1)

            
            
        return max_length
    

 *****************************************************
Koko eating bananas

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:

        n=len(piles)
        max_k =max(piles)
        left =1
        right =max_k


        while left < right :
            mid = (left + right)//2
            result = [math.ceil(x / mid) for x in piles]
            result =sum(result)
            if result <= h :
                right =mid

                
            if result > h :
                left = mid + 1
        
        return left 

***********************************************************************************

Majority Element :

n =len(nums)

count =Counter(nums)

for key,value in count.items() :
            if value >= n/2:
                return key


Without using counter :
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        
        n =len(nums)

        count ={}

        for num in nums :
            if num in count:
                count[num] +=1
            else :
                count[num] =1
            
        
        for key,value in count.items() :
            if value >= n/2:
                return key

Boyer-Moore Voting Algorithm (O(n) Time, O(1) Space) :

 res,count=None,0

        for num in nums :
            if count ==0:
                res=num
            count+=(1 if num==res  else -1)
        return res
    

**********************************************************

Reverse a linked list :

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        prev=None

        curr=head
        
        while curr:

            next_node =curr.next
            curr.next =prev
            prev =curr 
            curr=next_node
            
        return prev
        
 Merge two sorted list :
 
 
 # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:

        if not list1:  # If list1 is empty, return list2
            return list2
        if not list2:  # If list2 is empty, return list1
            return list1

        dummy=ListNode()
        current=dummy

        while list1 and list2:
            if list1.val <list2.val :
                current.next=list1
                list1 = list1.next
            else :
                current.next=list2
                list2 = list2.next
            current=current.next
            current.next = list1 if list1 else list2
        return dummy.next
        


************************************************************

19. Remove Nth Node From End of List

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        dummy =ListNode(0,head)
        slow=dummy
        fast=dummy

        for _ in range(n+1):
            fast=fast.next
        
        while fast :
            slow=slow.next
            fast=fast.next
        
        slow.next=slow.next.next

        return dummy.next
********************************************************************
141. Linked List Cycle

Floyd’s Cycle Detection Algorithm (also known as the Tortoise and Hare algorithm).
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:

        slow=head

        fast=head

        while fast and fast.next :
            slow=slow.next
            fast=fast.next.next

            if slow==fast:
                return True
        return False


  ***********************************************
160. Intersection of Two Linked Lists

Best Method

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:

        l1=headA
        l2=headB

        while l1 != l2 :
            l1=l1.next if l1 else headB
            l2=l2.next if l2 else headA
        return l1
  

Anoerth way hash set :


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:

        intersect = set()

        while headA:
            intersect.add(headA)
            headA=headA.next

        while headB :
            if headB in intersect:
                return headB
            headB=headB.next
            
        return None

 ******************************************

104. Maximum Depth of Binary Tree  
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:

        if root is None :
            return 0
        
        
        height_left=self.maxDepth(root.left)
        height_right=self.maxDepth(root.right)

        return max(height_left,height_right) + 1

        
       

        