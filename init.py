def main():
    while True:
        try:
            user_input = input("Enter something (Ctrl+C to exit): ")
            print(f"You entered: {user_input}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
