package Mih.demo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class ComponentB {
    @Autowired
    ComponentA componentA;
}
