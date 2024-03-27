import java.rmi.*;

public interface CalInterface extends Remote{
    public int add(int x, int y) throws RemoteException;
}
