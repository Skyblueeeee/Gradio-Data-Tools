import multiprocessing
import time

def A():
    print("Function A started")
    while True:
        time.sleep(1)
        print("Function A running...")

def B(process_a):
    print("Function B started")
    process_a.terminate()
    print("Function A terminated")

if __name__ == "__main__":
    process_a = multiprocessing.Process(target=A)
    process_a.start()

    while True:
        try:
            choice = int(input("Enter 1 to execute function A, 2 to kill function A: "))
            if choice == 1:
                if not process_a.is_alive():
                    process_a = multiprocessing.Process(target=A)
                    process_a.start()
                else:
                    print("Function A is already running.")
            elif choice == 2:
                B(process_a)
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
