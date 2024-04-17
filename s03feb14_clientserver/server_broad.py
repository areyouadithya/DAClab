import socket
import threading
import time
# Server code
def handle_client(client_socket, address):
    while True:
        try:
            # Receive data from the client
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break  # If no data received, break the loop
            print(f"Received from {address}: {data}")

        except ConnectionResetError:
            break

    remove_client(client_socket)

def send_hello_to_all_clients():
    while True:
        time.sleep(5)
        for client in clients:
            try:
                client.send("Hello to all Clients".encode('utf-8'))
            except ConnectionResetError:
                remove_client(client)
        time.sleep(2)  # Send the message every 5 seconds (adjust as needed)

def remove_client(client_socket):
    clients.remove(client_socket)
    client_socket.close()

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5555))
    server_socket.listen(5)
    print("Server listening on port 5555...")

    hello_thread = threading.Thread(target=send_hello_to_all_clients)
    hello_thread.start()

    while True:
        client_socket, address = server_socket.accept()
        clients.append(client_socket)
        client_thread = threading.Thread(target=handle_client, args=(client_socket, address))
        client_thread.start()

clients = []

start_server()