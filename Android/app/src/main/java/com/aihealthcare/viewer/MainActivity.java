package com.aihealthcare.viewer;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.net.DhcpInfo;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.util.Log;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.WindowManager;
import android.webkit.ConsoleMessage;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final String TAG = "AIHealthcareViewer";
    private static final String FALLBACK_SERVER_HOST = "192.168.137.1";
    private static final String EMULATOR_SERVER_HOST = "10.0.2.2";
    private static final int DEFAULT_SERVER_PORT = 8080;
    private static final int REQUESTED_TRACKS = 1;
    private static final int HEALTHCHECK_TIMEOUT_MS = 1500;

    private WebView viewerWebView;
    private String activeServerUrl;
    private Thread connectionThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        enableImmersiveMode();

        viewerWebView = findViewById(R.id.viewerWebView);

        configureWebView();
        connectToServer();
    }

    @Override
    protected void onResume() {
        super.onResume();
        enableImmersiveMode();
        String latestServerUrl = resolveServerUrl();
        if (activeServerUrl == null || !activeServerUrl.equals(latestServerUrl)) {
            connectToServer();
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            enableImmersiveMode();
        }
    }

    @SuppressLint({"SetJavaScriptEnabled", "AddJavascriptInterface"})
    private void configureWebView() {
        viewerWebView.getSettings().setJavaScriptEnabled(true);
        viewerWebView.getSettings().setDomStorageEnabled(true);
        viewerWebView.getSettings().setMediaPlaybackRequiresUserGesture(false);
        viewerWebView.getSettings().setAllowFileAccess(true);
        viewerWebView.getSettings().setAllowContentAccess(true);
        viewerWebView.getSettings().setAllowFileAccessFromFileURLs(true);
        viewerWebView.getSettings().setAllowUniversalAccessFromFileURLs(true);

        viewerWebView.addJavascriptInterface(new ViewerBridge(), "AndroidBridge");
        viewerWebView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onConsoleMessage(ConsoleMessage consoleMessage) {
                Log.d(TAG, "WebRTC console: " + consoleMessage.message());
                return super.onConsoleMessage(consoleMessage);
            }
        });
        viewerWebView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                showStatus("Viewer page loaded");
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                showStatus("Viewer load error: " + error.getDescription());
            }
        });

        if (BuildConfig.DEBUG) {
            WebView.setWebContentsDebuggingEnabled(true);
        }
    }

    private void connectToServer() {
        if (connectionThread != null && connectionThread.isAlive()) {
            connectionThread.interrupt();
        }

        connectionThread = new Thread(() -> {
            List<String> candidates = buildCandidateServerUrls();
            String selectedServerUrl = null;

            for (String candidate : candidates) {
                if (Thread.currentThread().isInterrupted()) {
                    return;
                }

                showStatus("Checking server " + candidate);
                if (isServerReachable(candidate)) {
                    selectedServerUrl = candidate;
                    break;
                }
            }

            final String finalServerUrl = selectedServerUrl != null
                    ? selectedServerUrl
                    : candidates.get(0);

            activeServerUrl = finalServerUrl;
            runOnUiThread(() -> loadViewer(finalServerUrl));
        }, "server-connector");
        connectionThread.start();
    }

    private void disconnectFromServer() {
        viewerWebView.evaluateJavascript(
                "window.viewerApp && window.viewerApp.shutdown && window.viewerApp.shutdown();",
                null
        );
        showStatus("Connection closed");
    }

    private String resolveServerUrl() {
        String gatewayIp = resolveGatewayIpAddress();
        if (gatewayIp == null || gatewayIp.isEmpty()) {
            gatewayIp = FALLBACK_SERVER_HOST;
        }
        return String.format(Locale.US, "http://%s:%d", gatewayIp, DEFAULT_SERVER_PORT);
    }

    private List<String> buildCandidateServerUrls() {
        LinkedHashSet<String> candidates = new LinkedHashSet<>();
        String gatewayServerUrl = resolveServerUrl();
        candidates.add(gatewayServerUrl);
        candidates.add(String.format(Locale.US, "http://%s:%d", FALLBACK_SERVER_HOST, DEFAULT_SERVER_PORT));
        candidates.add(String.format(Locale.US, "http://%s:%d", EMULATOR_SERVER_HOST, DEFAULT_SERVER_PORT));
        return new ArrayList<>(candidates);
    }

    private boolean isServerReachable(String serverUrl) {
        HttpURLConnection connection = null;
        try {
            URL url = new URL(serverUrl + "/health");
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(HEALTHCHECK_TIMEOUT_MS);
            connection.setReadTimeout(HEALTHCHECK_TIMEOUT_MS);
            connection.setUseCaches(false);
            int responseCode = connection.getResponseCode();
            Log.d(TAG, "Health check for " + serverUrl + " => " + responseCode);
            return responseCode >= 200 && responseCode < 300;
        } catch (IOException error) {
            Log.d(TAG, "Health check failed for " + serverUrl, error);
            return false;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private void loadViewer(String serverUrl) {
        showStatus("Connecting automatically to " + serverUrl);

        try {
            viewerWebView.clearCache(true);
            String viewerHtml = readViewerHtmlTemplate();
            String bootstrapScript = String.format(
                    Locale.US,
                    "<script>window.__VIEWER_CONFIG__={serverUrl:'%s',tracks:%d,assetBaseUrl:'file:///android_asset/'};</script>",
                    escapeForJavaScript(serverUrl),
                    REQUESTED_TRACKS
            );
            String hydratedHtml = viewerHtml.replace("<head>", "<head>" + bootstrapScript);
            viewerWebView.loadDataWithBaseURL(
                    serverUrl + "/",
                    hydratedHtml,
                    "text/html",
                    "utf-8",
                    null
            );
        } catch (IOException error) {
            Log.e(TAG, "Failed to load viewer template", error);
            showStatus("Viewer template load failed");
        }
    }

    private String resolveGatewayIpAddress() {
        WifiManager wifiManager = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wifiManager == null) {
            return null;
        }

        DhcpInfo dhcpInfo = wifiManager.getDhcpInfo();
        if (dhcpInfo == null || dhcpInfo.gateway == 0) {
            return null;
        }

        return String.format(
                Locale.US,
                "%d.%d.%d.%d",
                dhcpInfo.gateway & 0xff,
                (dhcpInfo.gateway >> 8) & 0xff,
                (dhcpInfo.gateway >> 16) & 0xff,
                (dhcpInfo.gateway >> 24) & 0xff
        );
    }

    private void showStatus(String message) {
        Log.d(TAG, message);
    }

    private void enableImmersiveMode() {
        final View decorView = getWindow().getDecorView();
        final int immersiveFlags =
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;

        decorView.setSystemUiVisibility(immersiveFlags);
        decorView.setOnSystemUiVisibilityChangeListener(visibility -> {
            if ((visibility & View.SYSTEM_UI_FLAG_FULLSCREEN) == 0) {
                decorView.post(() -> decorView.setSystemUiVisibility(immersiveFlags));
            }
        });
    }

    private String readViewerHtmlTemplate() throws IOException {
        try (InputStream inputStream = getAssets().open("viewer.html");
             ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int readCount;
            while ((readCount = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, readCount);
            }
            return outputStream.toString("UTF-8");
        }
    }

    private String escapeForJavaScript(String value) {
        return value
                .replace("\\", "\\\\")
                .replace("'", "\\'");
    }

    private String getDisplayInfoJson() {
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getRealMetrics(metrics);

        int widthPixels = metrics.widthPixels;
        int heightPixels = metrics.heightPixels;
        float widthDp = widthPixels / metrics.density;
        float heightDp = heightPixels / metrics.density;
        double widthInches = widthPixels / metrics.xdpi;
        double heightInches = heightPixels / metrics.ydpi;
        double diagonalInches = Math.sqrt((widthInches * widthInches) + (heightInches * heightInches));

        return String.format(
                Locale.US,
                "{\"resolution\":\"%d x %d px\",\"size\":\"%.2f in\",\"viewport\":\"%.0f x %.0f dp\",\"density\":\"%.0f dpi\"}",
                widthPixels,
                heightPixels,
                diagonalInches,
                widthDp,
                heightDp,
                metrics.densityDpi * 1.0f
        );
    }

    @Override
    protected void onDestroy() {
        if (connectionThread != null) {
            connectionThread.interrupt();
        }
        disconnectFromServer();
        viewerWebView.removeJavascriptInterface("AndroidBridge");
        viewerWebView.destroy();
        super.onDestroy();
    }

    private final class ViewerBridge {
        @JavascriptInterface
        public void postStatus(String message) {
            showStatus(message);
        }

        @JavascriptInterface
        public void postError(String message) {
            Log.e(TAG, message);
        }

        @JavascriptInterface
        public String getDisplayInfo() {
            return getDisplayInfoJson();
        }
    }
}
