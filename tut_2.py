# Base class
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"Hi, I'm {self.name} and I'm {self.age} years old.")

# Derived class
class Student(Person):
    def __init__(self, name, age, grade):
        super().__init__(name, age)  # call parent constructor
        self.grade = grade

    def introduce(self):
        super().introduce()  # optional: call base class method
        print(f"I'm in grade {self.grade}.")

# Create objects
person1 = Person("Alice", 30)
student1 = Student("Bob", 15, 10)

# Method calls
person1.introduce()
student1.introduce()
