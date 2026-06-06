"""

empty

"""

#
# def iszero(num):
#     return num == 0
#
#
# startingdial = 50
# counter = 0
#
# # Using a standard loop to handle inputs safely
# while True:
#     try:
#         inputs = input()
#         if not inputs:  # Break if input is empty
#             break
#     except EOFError:
#         break
#
#     direction = inputs[0]
#     ticks = int(inputs[1:])  # Extracts everything after the first letter
#
#     if direction == "L":
#         print("DOING LEFT")
#         for i in range(ticks):
#             startingdial -= 1  # Turn the dial down
#             if startingdial < 0:  # Wrap around if it goes below 0
#                 startingdial = 99
#
#             if iszero(startingdial):
#                 counter += 1
#
#     elif direction == "R":
#         print("DOING RIGHT")
#         for i in range(ticks):
#             startingdial += 1  # Turn the dial up
#             if startingdial > 99:  # Wrap around if it goes above 99
#                 startingdial = 0
#
#             if iszero(startingdial):
#                 counter += 1
#
# print(counter)


# def insertsort(array, size):
#     for i in range(size):
#         key = array[i]
#         j = i - 1
#         while j >= 0 and array[j] > key:
#             array[j + 1] = array[j]
#             j -= 1
#
#         array[j + 1] = key
#     return array
#
# list = [5,2,3,4,1]
# print(insertsort(list, len(list)))


# user_input = input("Enter brackets: ")
# array = []
#
# # Using a flag variable makes it easier to track if we found an error inside the loop
# is_valid = True
#
# for i in range(len(user_input)):
#     current_char = user_input[i]
#
#     # 1. Push opening brackets
#     if current_char == '(' or current_char == '[' or current_char == '{':
#         array.append(current_char)
#
#     # 2. Handle closing brackets
#     elif current_char == ')' or current_char == ']' or current_char == '}':
#
#         # Check to prevent IndexError: if stack is empty but we have a closing bracket, it's invalid
#         if not array:
#             is_valid = False
#             break
#
#         top_of_stack = array[-1]
#
#         # Check if the specific closing bracket matches the top opening bracket
#         if (current_char == ')' and top_of_stack == '(') or \
#                 (current_char == ']' and top_of_stack == '[') or \
#                 (current_char == '}' and top_of_stack == '{'):
#             array.pop()  # It's a match, take it off the stack!
#         else:
#             is_valid = False  # Mismatch found
#             break
#
# # 3. Final Check
# # If we didn't trigger 'is_valid = False' AND the stack is totally empty, we succeed!
# if is_valid and not array:
#     print(True)
# else:
#     print(False)

# cities = int(input())


# print(abs(1-2)/max(1,2))
# print(abs(1-3)/max(1,3))
# print(abs(2-4)/max(1,4))


