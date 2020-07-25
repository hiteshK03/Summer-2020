import socket
import threading
import sys

class Client:
	def __init__(self, host, port):

		self.s = socket.socket()

		try:
			self.s.connect((host, port))	
		except:
			print("Could not connect to server")
			if __name__ == '__main__':
				sys.exit()
		
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
				msg = self.s.recv(1024).decode()
				print("Server : ", msg)
			
			except:
				print("Err! Could not receive msg")
				break

			if msg == "Bye client!":
				self.s.close()
				break

	def sender(self):

		while self.t1.is_alive():
			msg = input()
			try:
				self.s.send(msg.encode())
				
			except:
				print("Err! could not send msg")
				break

			if msg == "Bye server!":
				self.s.close()
				break

if __name__ == '__main__':

	host = input("Enter host : ")
	port = int(input("Enter port : "))

	cli = Client(host, port)