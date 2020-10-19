package Mih.demo.Modules;

import org.apache.ibatis.type.Alias;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.Serializable;
import java.util.Date;

public class Student implements InitializingBean, DisposableBean, Serializable {

    private String studentId;

    private String name;

    private Date birthday;

    private String sex;

    public Student() {
        System.out.println("执行构造方法.");
    }

    public String getStudentId() {
        return studentId;
    }

    public void setStudentId(String studentId) {
        this.studentId = studentId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Date getBirthday() {
        return birthday;
    }

    public void setBirthday(Date birthday) {
        this.birthday = birthday;
    }

    public String getSex() {
        return sex;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    @Override
    public String toString() {
        return "number: " + studentId + "\n" +
                "name: " + name + "\n" +
                "birthday: " + birthday + "\n" +
                "sex: " + sex;
    }

    @PostConstruct
    public void testPostConstruct() {
        System.out.println("执行 @PostConstruct 方法.");
    }

    @PreDestroy
    public void testPreDestory() {
        System.out.println("执行 @PreDestroy 方法.");
    }



    @Override
    public void destroy() throws Exception {
        System.out.println("执行 DisposableBean 中的 destory().");
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        System.out.println("执行InitializingBean 中的 afterPropertiesSet().");
    }

    public void myInit() {
        System.out.println("执行指定的 init().");
    }

    public void myDestroy() {
        System.out.println("执行指定的 destroy().");
    }

}
