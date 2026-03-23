package com.aihealthcare.viewer;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.net.DhcpInfo;
import android.net.Uri;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.webkit.ConsoleMessage;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.util.Locale;

public class MainActivity extends Activity {
    private static final String TAG = "AIHealthcareViewer";
    private static final String FALLBACK_SERVER_HOST = "192.168.137.1";
    private static final int DEFAULT_SERVER_PORT = 8080;
    private static final int REQUESTED_TRACKS = 1;

    private WebView viewerWebView;
    private String activeServerUrl;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        viewerWebView = findViewById(R.id.viewerWebView);

        configureWebView();
        connectToServer();
    }

    @Override
    protected void onResume() {
        super.onResume();
        String latestServerUrl = resolveServerUrl();
        if (activeServerUrl == null || !activeServerUrl.equals(latestServerUrl)) {
            connectToServer();
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
        });

        if (BuildConfig.DEBUG) {
            WebView.setWebContentsDebuggingEnabled(true);
        }
    }

    private void connectToServer() {
        activeServerUrl = resolveServerUrl();

        String viewerUrl = String.format(
                Locale.US,
                "file:///android_asset/viewer.html?serverUrl=%s&tracks=%d",
                Uri.encode(activeServerUrl),
                REQUESTED_TRACKS
        );

        showStatus("Connecting automatically to " + activeServerUrl);
        viewerWebView.loadUrl(viewerUrl);
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

    @Override
    protected void onDestroy() {
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
    }
}
