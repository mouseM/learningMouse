package Mih.demo.Modules;

import java.io.Serializable;

public class Request<T extends Object> implements Serializable {
    private T data;

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }
}
