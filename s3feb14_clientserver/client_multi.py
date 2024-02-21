import socket

def client():
    host = socket.gethostname()
    port = 21042
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    try:
        while True:
            message = input("Enter your message (Type 'bye' to exit): ")
            if message.lower().strip() == "bye":
                break
            client_socket.send(message.encode())
    finally:
        client_socket.close()
client()