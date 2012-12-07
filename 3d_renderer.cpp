#include <stdio.h>
#include <iostream>
#include <winsock.h>
#include <winsock2.h>

using namespace std;
//#include <sys/types.h>
//#include <stdio.h>
//#include <unistd.h>
//#include <errno.h>
//#include <string.h>
//#include <stdlib.h>

int main()
{
	char pkt[] = { 'H', 'e', 'l', 'l', 'o' };
	int pkt_length = 5;
	char RecvBuf[1024];
	int BufLen = 1024;

	sockaddr_in dest;
	sockaddr_in local;
	sockaddr_in SenderAddr;
	int SenderAddrSize = sizeof (SenderAddr);

	WSAData data;
	WSAStartup( MAKEWORD( 2, 2 ), &data );

	local.sin_family = AF_INET;
	local.sin_addr.s_addr = inet_addr( "127.0.0.1" );
	local.sin_port = 12347; // choose any

	dest.sin_family = AF_INET;
	dest.sin_addr.s_addr = inet_addr( "127.0.0.1" );
	dest.sin_port = htons( 12345 );

	// create the socket
	SOCKET s = socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );
	// bind to the local address
	bind( s, (sockaddr *)&local, sizeof(local) );
	// send the pkt

	while(1){

		int ret = sendto( s, pkt, pkt_length, 0, (sockaddr *)&dest, sizeof(dest) );
		if(ret==pkt_length){
			cout<<"Sent packet\n";
		}else{
			cout<<"Failed to send\n";
		}

		int iResult = recv(s,RecvBuf, BufLen, 1);
		//int iResult = recvfrom( s, RecvBuf, sizeof( RecvBuf ), 1, (sockaddr *)&dest, 0 );
		cout<<"Bytes received: "<<iResult<<"\n";
		//cout<<RecvBuf<<"\n";
	}

}
