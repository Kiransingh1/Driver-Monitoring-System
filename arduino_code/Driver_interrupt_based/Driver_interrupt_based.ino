#include <avr/interrupt.h>
#include <avr/io.h>
char temp;
void setup()
{
   pinMode(13, OUTPUT);  // configuring pin 13 as output
   UBRR0 = 103; // for configuring baud rate of 9600bps
   UCSR0C |= (1 << UCSZ01) | (1 << UCSZ00); 
// Use 8-bit character sizes
   UCSR0B |= (1 << RXEN0) | (1 << TXEN0) | (1 << RXCIE0);  
// Turn on the transmission, reception, and Receive interrupt     
   sei();// enable global interrupt
}

void loop()
{
 
  switch(temp)
  {
    case 'a':
    digitalWrite(13,HIGH);
    delay(100);
    digitalWrite(13,LOW);
    delay(100);
    digitalWrite(13,HIGH);
    delay(100);
    digitalWrite(13,LOW);
    delay(100);
    break;

    case 'x':
    digitalWrite(13,LOW);
    break;

    case 'b':
    digitalWrite(13,HIGH);
    delay(500);
    digitalWrite(13,LOW);
    delay(500);
    digitalWrite(13,HIGH);
    delay(500);
    digitalWrite(13,LOW);
    delay(500);
    break;

    case 'c':
    digitalWrite(13,HIGH);
    delay(1000);
    digitalWrite(13,LOW);
    delay(1000);
    digitalWrite(13,HIGH);
    delay(1000);
    digitalWrite(13,LOW);
    delay(1000);
    break;

   
  }
}

ISR(USART_RX_vect)
{ 
  temp=UDR0;// read the received data byte in temp
}
