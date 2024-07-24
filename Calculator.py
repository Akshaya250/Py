def main():
    print("WE CAN DO FOLLOWING OPERATION: + - * / ")
    num1 = int(input("Enter num1: "))
    num2 = int(input("Enter num2: "))

    while True:
        operation = input("Enter operation + - * / ")

        if operation == "+":
            print(num1 + num2)
        elif operation == "-":
            print(num1 - num2)
        elif operation == "*":
            print(num1 * num2)
        elif operation == "/":
            print(num1 / num2)
        else:
            print("Please choose a correct operation")

        another_operation = input("Do you want to perform another operation? (yes/no): ")
        if another_operation.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
