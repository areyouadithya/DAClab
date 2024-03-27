import java.rmi.server.*;
public class CalClass extends UnicastRemoteObject implements CalInterface{
    
    public CalClass() throws Exception{
        super();
    }
    
    public int add(int x, int y){
        System.out.println("Received Request is "+ x + "+" + y);
        return x+y;
    }
}