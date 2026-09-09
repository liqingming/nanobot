package dev.nanobot.rider;

import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.Path;

/** 对真实 Python 本机服务运行 Java 客户端，不模拟 HTTP。 */
public final class BridgeSmoke {
    public static void main(String[] args) throws Exception {
        BridgeClient client = new BridgeClient();
        Path root = Path.of(args[1]);
        var targets = client.discover(Path.of(args[0]), root, root.resolve("玩家.cs"));
        if (targets.size() != 1) throw new AssertionError("会话发现数量错误：" + targets.size());
        var target = targets.getFirst();
        if (target.toString().contains(target.token())) throw new AssertionError("标签泄露令牌");
        JsonObject snapshot = new JsonObject();
        snapshot.addProperty("request_id", "java-smoke");
        snapshot.addProperty("path", root.resolve("玩家.cs").toString());
        snapshot.addProperty("start_line", 3);
        snapshot.addProperty("end_line", 4);
        snapshot.addProperty("text", "未保存的中文代码\nConsole.WriteLine(1);");
        client.send(target, snapshot);
        client.send(target, snapshot); // 重传不可重复添加。
        var stale = new BridgeClient.Target(target.port(), target.token(), target.instanceId(),
            target.workspace(), target.sessionKey(), "stale", target.title(), 0);
        try {
            client.send(stale, snapshot);
            throw new AssertionError("过期话题未拒绝");
        } catch (IOException expected) {
            if (!expected.getMessage().contains("重新添加")) throw expected;
        }
        if (!client.discover(Path.of(args[0]), root.resolve("other"), root.resolve("玩家.cs")).isEmpty())
            throw new AssertionError("跨项目目标未过滤");
        System.out.println("Java ↔ Python：会话发现、中文快照、重传去重、过期路由与项目隔离通过");
    }
}
