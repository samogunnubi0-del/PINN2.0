import fs from "node:fs";


const targets = await fetch("http://127.0.0.1:9333/json/list").then((response) => response.json());
const target =
  targets.find((item) => item.type === "page" && item.url === "about:blank") ||
  targets.find((item) => item.type === "page");
if (!target) throw new Error("No Edge page target found");

const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let nextId = 0;
const pending = new Map();
const events = [];
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
    return;
  }
  events.push(message);
});

function send(method, params = {}) {
  const id = ++nextId;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", {
  width: 390,
  height: 844,
  deviceScaleFactor: 1,
  mobile: true,
  screenWidth: 390,
  screenHeight: 844,
});
await send("Page.navigate", {
  url: "file:///C:/Users/ogunn/Downloads/New%20folder/Ac225_Expert_Email_Templates_2026.html",
});

await new Promise((resolve) => setTimeout(resolve, 700));

const result = await send("Runtime.evaluate", {
  expression: `JSON.stringify({
    innerWidth: window.innerWidth,
    bodyScrollWidth: document.body.scrollWidth,
    documentScrollWidth: document.documentElement.scrollWidth,
    pageRight: document.querySelector('.page').getBoundingClientRect().right,
    actionsRight: document.querySelector('.actions').getBoundingClientRect().right,
    widestRight: Math.max(...Array.from(document.querySelectorAll('body *')).map((node) => node.getBoundingClientRect().right)),
    narrowestLeft: Math.min(...Array.from(document.querySelectorAll('body *')).map((node) => node.getBoundingClientRect().left))
  })`,
  returnByValue: true,
});

const screenshot = await send("Page.captureScreenshot", {
  format: "png",
  fromSurface: true,
  captureBeyondViewport: false,
});
fs.writeFileSync(
  "C:/Users/ogunn/Downloads/New folder/tmp/pdfs/personalized_email_kit_mobile_cdp.png",
  Buffer.from(screenshot.data, "base64"),
);
console.log(result.result.value);
socket.close();
