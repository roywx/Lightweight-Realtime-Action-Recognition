import socket
import json

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))

    print("Connected to server")

    buffer = ""
    while True:
        data = client.recv(1024).decode('utf-8')
        if not data:
            break
        
        buffer += data
        
        # Process complete messages (split by newline)
        while '\n' in buffer:
            message, buffer = buffer.split('\n', 1)
            if message:
                action_data = json.loads(message)
                print(f"Action: {action_data['action']}, Confidence: {action_data['confidence']}")

if __name__ == "__main__":
    main()
