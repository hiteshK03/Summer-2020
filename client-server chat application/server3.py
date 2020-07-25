import socket
import threading

class Server:
	def __init__(self, port):
		self.host = ''
		self.port = port

		self.s = socket.socket()
		self.s.bind((self.host, self.port))

		print("socket binded to %s" %(port))
		
		self.s.listen(5)
		print("socket is listening")

		self.c, addr = self.s.accept()
		print("Got connection from", addr)

		self.t1 = threading.Thread(target = self.receiver)
		self.t2 = threading.Thread(target = self.sender)

		self.t1.start()
		self.t1.join()
		self.t2.join()

		self.s.close()

	def receiver(self):
		
		self.t2.start()

		while self.t2.is_alive():
			try:
				msg = self.c.recv(1024).decode()
				print("Server : ", msg)
				
			except:
				print("Err! Could not receive msg")
				break

			if msg == "Bye server!":
				self.c.close()
				break

	def sender(self):

		while self.t1.is_alive():
			msg = input()
			try:
				self.c.send(msg.encode())
				
			except:
				print("Err! Could not send msg")
				break

			if msg == "Bye client!":
				self.c.close()
				break
					
if __name__ == '__main__':

	port = int(input("Enter port : "))

	ser = Server(port)
			