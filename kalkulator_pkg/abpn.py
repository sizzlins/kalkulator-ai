# integer = input()
#
# def f(integer):
#     # Validate input lengthQ
#     if len(integer) < 7:
#         print("Valores nao aceitos")
#         return
#
#     # Try to parse the values with error handling
#     try:
#         # Check if characters at positions 0, 2, 4, 6 are digits
#         if (integer[0].isdigit() and integer[2].isdigit() and
#             integer[4].isdigit() and integer[6].isdigit()):
#             a, b, c, d = int(integer[0]), int(integer[2]), int(integer[4]), int(integer[6])
#         else:
#             # If not digits, try to split by spaces as fallback
#             parts = integer.split()
#             if len(parts) >= 4:
#                 a, b, c, d = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
#             else:
#                 print("Valores nao aceitos")
#                 return
#     except (ValueError, IndexError):
#         print("Valores nao aceitos")
#         return
#
#     # Check the conditions
#     if b > c and d > a and ((c+d) > (a+b)) and c >= 1 and d >= 1 and (a % 2 == 0):
#         print("Valores aceitos")
#     else:
#         print("Valores nao aceitos")
#
#
# f(integer)