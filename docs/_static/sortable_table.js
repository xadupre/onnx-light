/* Lightweight client-side enhancement: makes tables tagged with the
 * `onnx-light-sortable` CSS class sortable by clicking their headers,
 * and filterable via an associated search input.
 *
 * Markup contract:
 *   <input type="search" class="onnx-light-table-filter"
 *          data-table-target="my-table-id" placeholder="...">
 *   <table id="my-table-id" class="onnx-light-sortable">
 *     <thead><tr><th>...</th>...</tr></thead>
 *     <tbody>...</tbody>
 *   </table>
 *
 * Numeric columns are sorted numerically when every visible cell parses
 * as a number; otherwise sorting falls back to a case-insensitive string
 * comparison.
 */
(function () {
    "use strict";

    function cellValue(row, index) {
        var cell = row.cells[index];
        return cell ? cell.textContent.trim() : "";
    }

    function isNumeric(value) {
        if (value === "" || value === null || value === undefined) {
            return false;
        }
        return !isNaN(Number(value));
    }

    function compareFactory(index, numeric, ascending) {
        var direction = ascending ? 1 : -1;
        return function (rowA, rowB) {
            var a = cellValue(rowA, index);
            var b = cellValue(rowB, index);
            if (numeric) {
                return direction * (Number(a) - Number(b));
            }
            return direction * a.localeCompare(b, undefined, { sensitivity: "base" });
        };
    }

    function sortTable(table, headerCell, index) {
        var tbody = table.tBodies[0];
        if (!tbody) {
            return;
        }
        var rows = Array.prototype.slice.call(tbody.rows);
        if (rows.length === 0) {
            return;
        }
        var numeric = rows.every(function (row) {
            return isNumeric(cellValue(row, index));
        });
        var current = headerCell.getAttribute("data-sort-direction");
        var ascending = current !== "asc";
        rows.sort(compareFactory(index, numeric, ascending));

        // Reset indicators on sibling headers.
        var headers = headerCell.parentNode.cells;
        for (var i = 0; i < headers.length; i++) {
            headers[i].removeAttribute("data-sort-direction");
            headers[i].classList.remove("sort-asc", "sort-desc");
        }
        headerCell.setAttribute("data-sort-direction", ascending ? "asc" : "desc");
        headerCell.classList.add(ascending ? "sort-asc" : "sort-desc");

        var frag = document.createDocumentFragment();
        rows.forEach(function (row) {
            frag.appendChild(row);
        });
        tbody.appendChild(frag);
    }

    function enableSorting(table) {
        var headerRow = table.tHead && table.tHead.rows[0];
        if (!headerRow) {
            return;
        }
        for (var i = 0; i < headerRow.cells.length; i++) {
            (function (index) {
                var cell = headerRow.cells[index];
                cell.classList.add("sortable");
                cell.setAttribute("role", "button");
                cell.setAttribute("tabindex", "0");
                cell.addEventListener("click", function () {
                    sortTable(table, cell, index);
                });
                cell.addEventListener("keydown", function (event) {
                    if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        sortTable(table, cell, index);
                    }
                });
            })(i);
        }
    }

    function applyFilter(table, query) {
        var normalized = query.trim().toLowerCase();
        var tbody = table.tBodies[0];
        if (!tbody) {
            return;
        }
        for (var i = 0; i < tbody.rows.length; i++) {
            var row = tbody.rows[i];
            if (!normalized) {
                row.style.display = "";
                continue;
            }
            var text = row.textContent.toLowerCase();
            row.style.display = text.indexOf(normalized) !== -1 ? "" : "none";
        }
    }

    function enableFilters() {
        var inputs = document.querySelectorAll("input.onnx-light-table-filter");
        for (var i = 0; i < inputs.length; i++) {
            (function (input) {
                var targetId = input.getAttribute("data-table-target");
                if (!targetId) {
                    return;
                }
                var table = document.getElementById(targetId);
                if (!table) {
                    return;
                }
                input.addEventListener("input", function () {
                    applyFilter(table, input.value);
                });
            })(inputs[i]);
        }
    }

    function init() {
        var tables = document.querySelectorAll("table.onnx-light-sortable");
        for (var i = 0; i < tables.length; i++) {
            enableSorting(tables[i]);
        }
        enableFilters();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
