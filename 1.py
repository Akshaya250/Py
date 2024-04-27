'''x=5
a=6
y="0"
z="7"
print(type(x))
print(type(y))
print(x+a)
print(y+z)
a=int(z)
print(a+x)'''

'''l1 = [1, 2, 3, [8, 5.7, 6]]
l2 = l1[3][2]  # Accessing the third element of list l1, which is [8, 5.7, 6], then accessing the second element of that list.
print(l2)  # Output will be 6'''

'''t1=(2,8,6,9,5)
print(t1)'''

'''d1={'Name':'tanisha','Name2':'akshitha','Name3':'shraddha'}
print(d1)
print(type(d1))
print(d1.keys())
print(d1.values())
print(d1.items())
print(d1['Name'])
print(d1['Name2'])
print(d1['Name3'])'''

'''num1=int(input("enter a no: "))
num2=int(input("enter a no: "))
print(num1+num2)
print(num1*num2)'''

'''num=20
if(num>12):
    print("num is greater than 12")
else:
    print("num is lesser than 12")'''

'''def hello():
    print("hello")
    print("hello again")
hello()'''



def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Error! Division by zero."
    else:
        return x / y

print("Select operation:")
print("1. Addition")
print("2. Subtraction")
print("3. Multiplication")
print("4. Division")

while True:
    choice = input("Enter choice (1/2/3/4): ")

    if choice in ('1', '2', '3', '4'):
        try:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
        except ValueError:
            print("Invalid input")
            continue

        if choice == '1':
            print("Result:", add(num1, num2))
        elif choice == '2':
            print("Result:", subtract(num1, num2))
        elif choice == '3':
            print("Result:", multiply(num1, num2))
        elif choice == '4':
            print("Result:", divide(num1, num2))
    else:
        print("Invalid input")