#include <stdio.h>
#include <iostream>
#include <winsock.h>
#include <winsock2.h>

#ifdef WIN32
#include <sys/types.h>
#define    WINSOCKVERSION    MAKEWORD( 2,2 )
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#endif

#include <stdio.h>
#include <string.h>

#define MAX_BUFFER    128
#define HOST        "127.0.0.1"
#define PORT         8000

using namespace std;
//#include <sys/types.h>
//#include <stdio.h>
//#include <unistd.h>
//#include <errno.h>
//#include <string.h>
//#include <stdlib.h>

int main()
{
	//	char pkt[] = { 'H', 'e', 'l', 'l', 'o' };
	//	int pkt_length = 5;
	//	char RecvBuf[1024];
	//	int BufLen = 1024;
	//
	//	sockaddr_in dest;
	//	sockaddr_in local;
	//	sockaddr_in SenderAddr;
	//	int SenderAddrSize = sizeof (SenderAddr);
	//
	//	WSAData data;
	//	WSAStartup( MAKEWORD( 2, 2 ), &data );
	//
	//	local.sin_family = AF_INET;
	//	local.sin_addr.s_addr = inet_addr( "127.0.0.1" );
	//	local.sin_port = 12347; // choose any
	//
	//	dest.sin_family = AF_INET;
	//	dest.sin_addr.s_addr = inet_addr( "127.0.0.1" );
	//	dest.sin_port = htons( 12345 );
	//
	//	// create the socket
	//	SOCKET s = socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );
	//	// bind to the local address
	//	bind( s, (sockaddr *)&local, sizeof(local) );
	//	// send the pkt
	//
	//	while(1){
	//
	//		int ret = sendto( s, pkt, pkt_length, 0, (sockaddr *)&dest, sizeof(dest) );
	//		if(ret==pkt_length){
	//			cout<<"Sent packet\n";
	//		}else{
	//			cout<<"Failed to send\n";
	//		}
	//
	//		int iResult = recv(s,RecvBuf, BufLen, 1);
	//		//int iResult = recvfrom( s, RecvBuf, sizeof( RecvBuf ), 1, (sockaddr *)&dest, 0 );
	//		cout<<"Bytes received: "<<iResult<<"\n";
	//		//cout<<RecvBuf<<"\n";
	//	}
	// call_socket.c - A sample program to
	// demonstrate the TCP client
	//

	int connectionFd, rc, index = 0, limit = MAX_BUFFER;
	struct sockaddr_in servAddr, localAddr;
	char buffer[MAX_BUFFER+1];


	// Start up WinSock2
	WSADATA wsaData;
	if( WSAStartup( WINSOCKVERSION, &wsaData) != 0 )
		return ERROR;


	memset(&servAddr, 0, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(PORT);
	servAddr.sin_addr.s_addr = inet_addr(HOST);

	// Create socket
	connectionFd = socket(AF_INET, SOCK_STREAM, 0);

	/* bind any port number */
	localAddr.sin_family = AF_INET;
	localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	localAddr.sin_port = htons(9000);

	rc = bind(connectionFd,
			(struct sockaddr *) &localAddr, sizeof(localAddr));

	// Connect to Server
	connect(connectionFd,
			(struct sockaddr *)&servAddr, sizeof(servAddr));


	// Receive data from Server
	sprintf( buffer, "%s", "" );
	recv(connectionFd, buffer, MAX_BUFFER, 0);
	printf("Client read from Server [ %s ]\n", buffer);

	// Send request to Server
	sprintf( buffer, "%s", "Hello from opencv" );
	send( connectionFd, buffer, strlen(buffer), 0 );
	printf("Client sent to sever %s\n", buffer);

	closesocket(connectionFd);
	printf("Client closed.\n");

	return(0);

}
