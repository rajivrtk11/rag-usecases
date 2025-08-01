# num = int(input("Enter a number: "))
# print("Even" if num % 2 == 0 else "Odd")

# def factorial(n):
#     return 1 if n == 0 else n * factorial(n - 1)

# print(factorial(5))  # Output: 120
from tut_2 import Person, Student
from module.hello import hello
# a, b = 0, 1
# for _ in range(10):
#     print(a, end=" ")
#     a, b = b, a + b

# Create objects
person1 = Person("Alice", 30)
student1 = Student("Bob", 15, 10)

# Method calls
# person1.introduce()
# student1.introduce()
hello()  # Call the hello function from the module


