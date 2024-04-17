import java.rmi.*;
import java.rmi.registry.LocateRegistry;

public class Server
{
  public static void main (String[] argv) 
  {
    try 
    {
      CalInterface remoteObj = new CalClass();
      LocateRegistry.createRegistry(1920);
      Naming.rebind("rmi://localhost:1920"+"/calc",remoteObj);
      System.out.println ("Calculator Server is ready.");
    } 
    catch (Exception e) 
    {
      System.out.println ("Calculator Server failed: " + e);
    }
  }
}
