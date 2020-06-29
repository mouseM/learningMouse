package ActiveObjectsTest;

public class MakerClientThread extends Thread {

    private final ActiveObject activeObject;
    private final char c;

    public MakerClientThread(ActiveObject activeObject, String name) {
        super(name);
        this.activeObject = activeObject;
        this.c = name.charAt(0);
    }

    @Override
    public void run() {
        try {
            for (int i = 0; true; i ++) {
                Result result = activeObject.makeString(i + 1, this.c);
//                Thread.sleep(10);
                String value = (String) result.getResultValue();
                System.out.println(Thread.currentThread().getName() + ": value = " + value);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
