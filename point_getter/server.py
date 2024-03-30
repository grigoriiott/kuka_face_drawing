import socket
import sys
from time import sleep

from point_getter import *

def get_flag(flag):
    message = """<xml version="1.0" encoding="UTF-8">
                    <Command>
                        <Signal>
                            <STRT1>{}</STRT1>
                        </Signal>
                    </Command>""".format(flag)
    return message

def get_message_12(idx, lms):
    #print(idx+1)
    message = """<xml version="1.0" encoding="UTF-8">
                    <Command>
                        <Coords>
                            <PTX1>{}</PTX1><PTY1>{}</PTY1>
                            <PTX2>{}</PTX2><PTY2>{}</PTY2>
                            <PTX3>{}</PTX3><PTY3>{}</PTY3>
                            <PTX4>{}</PTX4><PTY4>{}</PTY4>
                            <PTX5>{}</PTX5><PTY5>{}</PTY5>
                            <PTX6>{}</PTX6><PTY6>{}</PTY6>
                            <PTX7>{}</PTX7><PTY7>{}</PTY7>
                            <PTX8>{}</PTX8><PTY8>{}</PTY8>
                            <PTX9>{}</PTX9><PTY9>{}</PTY9>
                            <PTX10>{}</PTX10><PTY10>{}</PTY10>
                            <PTX11>{}</PTX11><PTY11>{}</PTY11>
                            <PTX12>{}</PTX12><PTY12>{}</PTY12>
                        </Coords>
                    </Command>""".format(lms[idx+1][0][1], lms[idx+1][0][2], lms[idx+1][1][1], lms[idx+1][1][2], 
                                        lms[idx+1][2][1], lms[idx+1][2][2], lms[idx+1][3][1], lms[idx+1][3][2], 
                                        lms[idx+1][4][1], lms[idx+1][4][2], lms[idx+1][5][1], lms[idx+1][5][2], 
                                        lms[idx+1][6][1], lms[idx+1][6][2], lms[idx+1][7][1], lms[idx+1][7][2], 
                                        lms[idx+1][8][1], lms[idx+1][8][2], lms[idx+1][9][1], lms[idx+1][9][2], 
                                        lms[idx+1][10][1], lms[idx+1][10][2], lms[idx+1][11][1], lms[idx+1][11][2])
    return message

def get_message_3(idx, lms, flag):
    message = """<xml version="1.0" encoding="UTF-8">
                    <Command>
                        <Coords>
                            <PTX1>{}</PTX1><PTY1>{}</PTY1>
                            <PTX2>{}</PTX2><PTY2>{}</PTY2>
                            <PTX3>{}</PTX3><PTY3>{}</PTY3>
                        </Coords>
                        <Signal>
                            <STRT1>{}</STRT1>
                        </Signal>
                    </Command>""".format(lms[idx][0][1], lms[idx][0][0], lms[idx][1][1], 
                                         lms[idx][1][0], lms[idx][2][1], lms[idx][2][0],
                                         flag)
    return message

def get_message_3_short(idx, lms):
    message = """<xml version="1.0" encoding="UTF-8">
                    <Command>
                        <Coords>
                            <PTX1>{}</PTX1><PTY1>{}</PTY1>
                            <PTX2>{}</PTX2><PTY2>{}</PTY2>
                            <PTX3>{}</PTX3><PTY3>{}</PTY3>
                        </Coords>
                        <Signal>
                            <STRT1>{}</STRT1>
                        </Signal>
                    </Command>""".format(lms[idx][0][1], lms[idx][0][0], lms[idx][1][1], 
                                         lms[idx][1][0], lms[idx][2][1], lms[idx][2][0])
    return message

# <ELEMENT Tag="Command/Signal/STRT1" Type="BOOL"/>
#format(lms[idx+1][0][1], lms[idx+1][0][0], lms[idx+1][1][1], lms[idx+1][1][0], lms[idx+1][2][1], lms[idx+1][2][0])
#"192.168.88.1"
#point = {X 489, Y 412, Z 269, A 89, B 0, C -179}
def client_program():
    host = "192.168.88.172"
    port = 54610

    results = get_all_points()

    #print(get_message_12(len(results)-2, results))

    client_soc = socket.socket()
    client_soc.connect((host, port))
    #sleep(1)
    for i in range(len(results)-2):
        data = client_soc.recv(1024).decode( 'utf-8' )
        print("Received :", repr(data))
        if i == (len(results)-3):
            client_soc.send(get_flag(0).encode())
            message = get_message_12(i, results)
        else:
            client_soc.send(get_flag(1).encode())
            message = get_message_12(i, results)
        
        client_soc.send(message.encode())
        #print(message)
        #print("Message sent #", i+1)

    for i in range(len(results[0])):
        data = client_soc.recv(1024).decode( 'utf-8' )
        print("Received :", repr(data))
        print(i)
        if i == (len(results[0])-1):
            message = get_message_3(i, results[0], 0)
        else:
            message = get_message_3(i, results[0], 1)
        client_soc.send(message.encode())

    data = client_soc.recv(1024).decode( 'utf-8' )
    print("Received :", repr(data))
    message = get_message_12(len(results)-2, results)
    client_soc.send(message.encode())
    
    sleep(10)
    client_soc.close()

if __name__ == "__main__":
    client_program()
