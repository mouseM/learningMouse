package Mih.demo;

public class Test {

    public static void main(String[] args) {
        byte[] bytes = "mih".getBytes();
        for (byte b : bytes) {
            System.out.print(b + " ");
        }
        System.out.println();
        String str = new String(bytes);
        System.out.println(str);
    }
}
