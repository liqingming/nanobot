package dev.nanobot.rider;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.Proxy;
import java.net.ProxySelector;
import java.net.SocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/** 仅向当前用户登记的回环端点发送选区，不使用系统代理或重定向。 */
public final class BridgeClient {
    private final HttpClient http = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .connectTimeout(Duration.ofSeconds(2))
        .followRedirects(HttpClient.Redirect.NEVER)
        .proxy(new ProxySelector() {
            public List<Proxy> select(URI uri) { return List.of(Proxy.NO_PROXY); }
            public void connectFailed(URI uri, SocketAddress address, IOException error) { }
        }).build();

    public record Target(int port, String token, String instanceId, String workspace,
                         String sessionKey, String generation, String title, int pending) {
        @Override public String toString() {
            // 不允许凭证进入弹窗、日志或异常信息。
            return title + " [" + sessionKey + "] · " + instanceId.substring(0, 8)
                + " · " + workspace + " · 待发送 " + pending;
        }
    }

    public List<Target> discover(Path registry, Path project, Path selectedFile) throws IOException {
        List<Target> targets = new ArrayList<>();
        if (!Files.isDirectory(registry)) return targets;
        Path projectRoot = project.toRealPath();
        Path selected = selectedFile.toAbsolutePath().normalize();
        try (var files = Files.list(registry)) {
            for (Path file : files.filter(p -> p.getFileName().toString().endsWith(".json"))
                    .sorted().limit(128).toList()) {
                try {
                    if (Files.size(file) > 16384) continue;
                    JsonObject manifest = JsonParser.parseString(Files.readString(file)).getAsJsonObject();
                    if (manifest.get("version").getAsInt() != 1) continue;
                    Path workspace = Path.of(manifest.get("workspace").getAsString()).toRealPath();
                    if (!workspace.startsWith(projectRoot) || !selected.startsWith(workspace)) continue;
                    int port = manifest.get("port").getAsInt();
                    String token = manifest.get("token").getAsString();
                    String instance = manifest.get("instance_id").getAsString();
                    if (port < 1 || port > 65535 || !instance.matches("[a-f0-9]{32}")
                            || !token.matches("[A-Za-z0-9_-]{40,128}")) continue;
                    JsonObject live = request(port, token, "/v1/session", null);
                    if (!instance.equals(live.get("instance_id").getAsString())
                            || !workspace.equals(Path.of(live.get("workspace").getAsString()).toRealPath()))
                        continue;
                    targets.add(new Target(port, token, instance, workspace.toString(),
                        live.get("session_key").getAsString(), live.get("generation").getAsString(),
                        live.get("title").getAsString(), live.get("pending").getAsInt()));
                } catch (IOException | RuntimeException ignored) {
                    // 已退出进程的登记文件和不兼容端点不应阻断其他会话。
                }
            }
        }
        return targets;
    }

    public void send(Target target, JsonObject selection) throws IOException {
        JsonObject payload = selection.deepCopy();
        payload.addProperty("session_key", target.sessionKey());
        payload.addProperty("generation", target.generation());
        request(target.port(), target.token(), "/v1/context", payload);
    }

    private JsonObject request(int port, String token, String route, JsonObject body)
            throws IOException {
        HttpRequest.Builder request = HttpRequest.newBuilder(
            URI.create("http://127.0.0.1:" + port + route))
            .timeout(Duration.ofSeconds(3)).header("Authorization", "Bearer " + token);
        if (body == null) request.GET();
        else request.header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8));
        try {
            HttpResponse<String> response = http.send(request.build(),
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() != 200) {
                if (response.statusCode() == 409)
                    throw new IOException("目标话题或附件已更新，请重新添加并选择会话。");
                if (response.statusCode() == 401)
                    throw new IOException("终端接入凭证已更新，请重新添加。");
                throw new IOException("终端拒绝选区（HTTP " + response.statusCode()
                    + "）；请检查工作区、附件数量及大小。");
            }
            return JsonParser.parseString(response.body()).getAsJsonObject();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IOException("请求已取消。");
        }
    }
}
