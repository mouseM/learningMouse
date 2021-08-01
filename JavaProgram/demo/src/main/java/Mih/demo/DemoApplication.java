package Mih.demo;

import Mih.demo.Controllers.RestfulTest;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.ConfigurableApplicationContext;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;


@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {

		ConfigurableApplicationContext applicationContext = SpringApplication.run(DemoApplication.class, args);
		System.out.println("IOC 容器启动完成！");
//		RestfulTest lazyInitBean = applicationContext.getBean("restfulTest", RestfulTest.class);
//		if (lazyInitBean != null) {
//			System.out.println("懒加载的Bean已经加载！");
//		}

//		applicationContext.close();

	}
}
