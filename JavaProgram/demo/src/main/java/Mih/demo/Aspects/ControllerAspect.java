package Mih.demo.Aspects;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import javax.servlet.http.HttpServletRequest;
import java.util.Arrays;

@Aspect
@Component
public class ControllerAspect {

    private static final Logger LOGGER = LoggerFactory.getLogger(ControllerAspect.class);

    @Pointcut("execution(public * Mih.demo.Controllers.RestfulTest.hello(..)))")
    public void pointCut() { }

//    @Pointcut("execution(* Mih.demo.Controllers.*.*(..))")
//    public void LogPointCut() {}

    @Before("pointCut()")
    public void beforProcess(){
        System.out.println("before!");
    }

    @After("pointCut()")
    public void afterProcess() {
        System.out.println("after!");
    }

//    @Before("LogPointCut()")
//    public void doBefore(JoinPoint joinPoint) {
//        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
//        HttpServletRequest request = attributes.getRequest();
//        //url
//        LOGGER.info("url:{}", request.getRequestURL());
//        //method
//        LOGGER.info("method:{}", request.getMethod());
//        //ip
//        LOGGER.info("ip:{}", request.getRemoteAddr());
//        //类方法
//        LOGGER.info("class_method:{}", joinPoint.getSignature().getDeclaringTypeName() + "." + joinPoint.getSignature().getName());
//        //参数(joinPoint.getArgs()返回一个参数数组)
//        LOGGER.info("args: ");
//        Object[] args = joinPoint.getArgs();
//        Arrays.stream(args).forEach(arg -> {
//            LOGGER.info(arg.toString());
//        });
////        LOGGER.info("args:{}", Arrays.asList(joinPoint.getArgs()));
//    }
//
//    @AfterReturning(returning = "object", pointcut = "LogPointCut()")
//    public void doAfterReturning(Object object) {
//        LOGGER.info("response:{}", object.toString());
//    }

}
