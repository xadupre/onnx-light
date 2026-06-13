// Click-to-zoom for diagrams tagged with the "zoomable-svg" class.
// Opens the image in a full-screen overlay where the mouse wheel zooms and
// dragging pans, so large SVGs (such as the proto relations graph) can be
// inspected closely. No external dependencies are required.
(function () {
    "use strict";

    function createOverlay(src, alt) {
        var overlay = document.createElement("div");
        overlay.className = "svg-zoom-overlay";

        var stage = document.createElement("div");
        stage.className = "svg-zoom-stage";

        var image = document.createElement("img");
        image.src = src;
        image.alt = alt || "";
        image.className = "svg-zoom-image";

        var scale = 1;
        var translateX = 0;
        var translateY = 0;

        function applyTransform() {
            image.style.transform =
                "translate(" + translateX + "px, " + translateY + "px) scale(" + scale + ")";
        }

        stage.addEventListener("wheel", function (event) {
            event.preventDefault();
            var factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
            scale = Math.min(20, Math.max(0.2, scale * factor));
            applyTransform();
        });

        var dragging = false;
        var lastX = 0;
        var lastY = 0;

        stage.addEventListener("mousedown", function (event) {
            dragging = true;
            lastX = event.clientX;
            lastY = event.clientY;
            event.preventDefault();
        });

        function onMouseUp() {
            dragging = false;
        }

        function onMouseMove(event) {
            if (!dragging) {
                return;
            }
            translateX += event.clientX - lastX;
            translateY += event.clientY - lastY;
            lastX = event.clientX;
            lastY = event.clientY;
            applyTransform();
        }

        window.addEventListener("mouseup", onMouseUp);
        window.addEventListener("mousemove", onMouseMove);

        var hint = document.createElement("div");
        hint.className = "svg-zoom-hint";
        hint.textContent =
            "Scroll to zoom, drag or use arrow keys to pan, press Esc or click the background to close.";

        function onKeyDown(event) {
            if (event.key === "Escape") {
                close();
                return;
            }
            var step = 40;
            if (event.key === "ArrowLeft") {
                translateX += step;
            } else if (event.key === "ArrowRight") {
                translateX -= step;
            } else if (event.key === "ArrowUp") {
                translateY += step;
            } else if (event.key === "ArrowDown") {
                translateY -= step;
            } else {
                return;
            }
            event.preventDefault();
            applyTransform();
        }

        function close() {
            document.removeEventListener("keydown", onKeyDown);
            window.removeEventListener("mouseup", onMouseUp);
            window.removeEventListener("mousemove", onMouseMove);
            if (overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        }

        overlay.addEventListener("click", function (event) {
            if (event.target === overlay) {
                close();
            }
        });
        document.addEventListener("keydown", onKeyDown);

        stage.appendChild(image);
        overlay.appendChild(stage);
        overlay.appendChild(hint);
        applyTransform();
        return overlay;
    }

    function enableZoom(image) {
        image.style.cursor = "zoom-in";
        if (!image.title) {
            image.title = "Click to zoom";
        }
        image.addEventListener("click", function () {
            document.body.appendChild(createOverlay(image.src, image.alt));
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        var images = document.querySelectorAll("img.zoomable-svg");
        for (var i = 0; i < images.length; i++) {
            enableZoom(images[i]);
        }
    });
})();
