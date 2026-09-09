package dev.nanobot.rider;

import com.google.gson.JsonObject;
import com.intellij.notification.NotificationGroupManager;
import com.intellij.notification.NotificationType;
import com.intellij.openapi.actionSystem.ActionUpdateThread;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.project.DumbAwareAction;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.popup.JBPopupFactory;
import org.jetbrains.annotations.NotNull;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.UUID;
import java.util.function.Consumer;

/** 在 EDT 获取未保存的选区快照；发现与传送都在后台执行。 */
public final class AddSelectionAction extends DumbAwareAction {
    @Override public @NotNull ActionUpdateThread getActionUpdateThread() {
        return ActionUpdateThread.BGT;
    }

    @Override public void update(@NotNull AnActionEvent event) {
        Editor editor = event.getData(CommonDataKeys.EDITOR);
        event.getPresentation().setEnabledAndVisible(event.getProject() != null
            && editor != null && editor.getSelectionModel().hasSelection());
    }

    @Override public void actionPerformed(@NotNull AnActionEvent event) {
        Project project = event.getProject();
        Editor editor = event.getData(CommonDataKeys.EDITOR);
        if (project == null || editor == null || project.getBasePath() == null) return;
        var file = FileDocumentManager.getInstance().getFile(editor.getDocument());
        var selection = editor.getSelectionModel();
        String text = selection.getSelectedText();
        if (file == null || !file.isInLocalFileSystem() || text == null || text.isEmpty()) return;
        if (editor.getCaretModel().getCaretCount() != 1) {
            notify(project, "暂不支持多光标/矩形选区，请只选择一个连续代码段。", true);
            return;
        }
        if (text.getBytes(StandardCharsets.UTF_8).length > 64 * 1024) {
            notify(project, "单个选区不能超过 64 KiB，请缩小选区。", true);
            return;
        }
        JsonObject snapshot = new JsonObject();
        snapshot.addProperty("request_id", UUID.randomUUID().toString());
        snapshot.addProperty("path", file.getPath());
        snapshot.addProperty("text", text);
        int start = selection.getSelectionStart();
        int end = selection.getSelectionEnd();
        int firstLine = editor.getDocument().getLineNumber(start) + 1;
        // 选区终点是半开区间；恰好落在下一行行首时不多报一行。
        int lastLine = editor.getDocument().getLineNumber(Math.max(start, end - 1)) + 1;
        snapshot.addProperty("start_line", firstLine);
        snapshot.addProperty("end_line", lastLine);
        Path root = Path.of(project.getBasePath());
        Path selectedPath = Path.of(file.getPath());
        String label = file.getName() + ":" + firstLine + "–" + lastLine;
        ApplicationManager.getApplication().executeOnPooledThread(() -> {
            try {
                BridgeClient client = new BridgeClient();
                Path registry = Path.of(System.getProperty("user.home"), ".nanobot", "ide-bridge");
                var targets = client.discover(registry, root, selectedPath);
                ApplicationManager.getApplication().invokeLater(() -> {
                    if (project.isDisposed()) return;
                    if (targets.isEmpty()) {
                        notify(project, "未找到此项目的 nanobot 会话。请在项目目录启动 nanobot，"
                            + "输入 /ide on 后重试。", true);
                        return;
                    }
                    Consumer<BridgeClient.Target> send = target ->
                        ApplicationManager.getApplication().executeOnPooledThread(() -> {
                            try {
                                client.send(target, snapshot);
                                notify(project, "已添加 " + label + " 到「" + target.title()
                                    + "」。请在终端输入问题后发送。", false);
                            } catch (Exception error) {
                                notify(project, "添加失败：" + error.getMessage(), true);
                            }
                        });
                    // 唯一有效目标直接添加；仍使用发现时的代次，由接收端拒绝过期话题。
                    if (targets.size() == 1) {
                        send.accept(targets.getFirst());
                        return;
                    }
                    JBPopupFactory.getInstance().createPopupChooserBuilder(targets)
                        .setTitle("添加 " + label + " 到哪个 nanobot 会话？")
                        .setItemChosenCallback(send::accept)
                        .createPopup().showCenteredInCurrentWindow(project);
                });
            } catch (Exception error) {
                notify(project, "无法发现 nanobot 会话，请检查本机接入登记目录。", true);
            }
        });
    }

    private static void notify(Project project, String message, boolean error) {
        ApplicationManager.getApplication().invokeLater(() -> {
            if (!project.isDisposed())
                NotificationGroupManager.getInstance().getNotificationGroup("Nanobot Context")
                    .createNotification(message,
                        error ? NotificationType.WARNING : NotificationType.INFORMATION)
                    .notify(project);
        });
    }
}
