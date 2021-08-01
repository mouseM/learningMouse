package Mih.demo.TestBeans;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class B {

    private String b;

    @Autowired
    A a;

    public void setB(String b) {
        this.b = b;
    }

    public String getB() {
        return b;
    }
}
