(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutPanelUtils = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function escapeHtml(value) {
        if (value === undefined || value === null) return "";
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function escapeJsxDoubleQuotedString(value) {
        if (!value) return "";
        return String(value)
            .replace(/\\/g, "\\\\")
            .replace(/"/g, '\\"')
            .replace(/\n/g, "\\n")
            .replace(/\r/g, "\\r")
            .replace(/\t/g, "\\t");
    }

    function createLazyDomProxy(documentRef, cache) {
        var doc = documentRef || (typeof document !== "undefined" ? document : null);
        var target = cache || {};
        return new Proxy(target, {
            get: function (targetObj, id) {
                if (typeof id !== "string") return targetObj[id];
                if (id in targetObj) return targetObj[id];
                var node = doc && typeof doc.getElementById === "function"
                    ? doc.getElementById(id)
                    : null;
                if (node) targetObj[id] = node;
                return node;
            }
        });
    }

    function normalizePaletteText(value) {
        return (value || "").toLowerCase().replace(/\s+/g, " ").trim();
    }

    function formatPaletteLabel(value) {
        return (value || "").replace(/[-_]+/g, " ").replace(/\b[a-z]/g, function (letter) {
            return letter.toUpperCase();
        });
    }

    function getCommandPaletteItemKey(item) {
        if (!item) return "";
        return [item.name || "", item.tab || "", item.sub || ""].join("::");
    }

    function descriptionForItem(item, descriptionMap) {
        var itemKey = getCommandPaletteItemKey(item);
        if (descriptionMap && descriptionMap[itemKey]) return descriptionMap[itemKey];
        if (!item) return "Open tools across the editing workflow.";
        switch (item.tab) {
        case "cut":
            return "Tighten pacing, trims, and spoken edits from one focused cut workflow.";
        case "captions":
            return "Transcribe, translate, and shape subtitle deliverables without leaving the panel.";
        case "audio":
            return "Polish dialogue, stems, loudness, and generated sound from one audio surface.";
        case "video":
            return "Repair, reframe, and finish image work with cleaner visual controls.";
        case "export":
            return "Build deliverables, thumbnails, and repeatable output presets faster.";
        case "timeline":
            return "Write sequence edits and timeline metadata back into Premiere with more control.";
        case "nlp":
            return "Use search and language-driven tools to find footage or trigger edit actions.";
        case "settings":
            return "Adjust workspace defaults, templates, and system-level behavior.";
        default:
            return "Open this tool and jump directly to the matching workspace.";
        }
    }

    function makePaletteContext(options) {
        options = options || {};
        return {
            items: Array.isArray(options.items) ? options.items : [],
            query: normalizePaletteText(options.query),
            activeTab: options.activeTab || "",
            historyKeys: Array.isArray(options.historyKeys) ? options.historyKeys : [],
            favoriteIds: Array.isArray(options.favoriteIds) ? options.favoriteIds : [],
            descriptionMap: options.descriptionMap || {},
            getTabLabel: typeof options.getTabLabel === "function"
                ? options.getTabLabel
                : function (tab) { return formatPaletteLabel(tab); },
            getSubLabel: typeof options.getSubLabel === "function"
                ? options.getSubLabel
                : function (sub) { return formatPaletteLabel(sub); },
            getFavoriteId: typeof options.getFavoriteId === "function"
                ? options.getFavoriteId
                : function () { return ""; },
            getItemForFavorite: typeof options.getItemForFavorite === "function"
                ? options.getItemForFavorite
                : function () { return null; }
        };
    }

    function createPaletteEntry(item, ctx, extras) {
        extras = extras || {};
        var key = getCommandPaletteItemKey(item);
        var favoriteId = ctx.getFavoriteId(item) || "";
        var tabLabel = ctx.getTabLabel(item.tab);
        var subLabel = ctx.getSubLabel(item.sub);
        return {
            item: item,
            key: key,
            description: descriptionForItem(item, ctx.descriptionMap),
            tabLabel: tabLabel,
            subLabel: subLabel,
            location: subLabel ? (tabLabel + " / " + subLabel) : tabLabel,
            favoriteId: favoriteId,
            isFavorite: favoriteId ? ctx.favoriteIds.indexOf(favoriteId) !== -1 : false,
            isRecent: !!extras.isRecent,
            isCurrent: !!extras.isCurrent,
            score: extras.score || 0
        };
    }

    function scoreCommandPaletteItem(item, ctx) {
        var q = ctx.query;
        if (!q) return 0;

        var name = normalizePaletteText(item.name);
        var keywords = normalizePaletteText(item.keywords);
        var tabLabel = normalizePaletteText(ctx.getTabLabel(item.tab));
        var subLabel = normalizePaletteText(ctx.getSubLabel(item.sub));
        var score = 0;
        var matchedTokens = 0;
        var tokens = q.split(" ");
        var favoriteId = ctx.getFavoriteId(item);

        if (name === q) score += 220;
        else if (name.indexOf(q) === 0) score += 140;
        else if (name.indexOf(q) !== -1) score += 96;

        if (keywords.indexOf(q) !== -1) score += 56;
        if (tabLabel.indexOf(q) !== -1) score += 24;
        if (subLabel.indexOf(q) !== -1) score += 28;

        for (var i = 0; i < tokens.length; i++) {
            var token = tokens[i];
            if (!token) continue;
            if (name.indexOf(token) !== -1) {
                score += 32;
                matchedTokens++;
            } else if (keywords.indexOf(token) !== -1) {
                score += 18;
                matchedTokens++;
            } else if (tabLabel.indexOf(token) !== -1 || subLabel.indexOf(token) !== -1) {
                score += 10;
                matchedTokens++;
            }
        }

        if (!score && !matchedTokens) return 0;
        if (matchedTokens > 1) score += matchedTokens * 8;
        if (favoriteId && ctx.favoriteIds.indexOf(favoriteId) !== -1) score += 16;
        if (item.tab === ctx.activeTab) score += 12;
        return score;
    }

    function getItemByKey(items, key) {
        for (var i = 0; i < items.length; i++) {
            if (getCommandPaletteItemKey(items[i]) === key) return items[i];
        }
        return null;
    }

    function buildPaletteEntries(items, ctx, resolver, seen) {
        var entries = [];
        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            if (!item) continue;
            var key = getCommandPaletteItemKey(item);
            if (seen && seen[key]) continue;
            if (seen) seen[key] = true;
            entries.push(createPaletteEntry(item, ctx, resolver ? resolver(item, key, i) : null));
        }
        return entries;
    }

    function addPaletteSection(sections, label, items, ctx, resolver, seen) {
        var entries = buildPaletteEntries(items, ctx, resolver, seen);
        if (entries.length) sections.push({ label: label, entries: entries });
    }

    function buildCommandPaletteSections(options) {
        var ctx = makePaletteContext(options);
        var sections = [];
        var historyLookup = {};
        for (var i = 0; i < ctx.historyKeys.length; i++) historyLookup[ctx.historyKeys[i]] = true;

        if (!ctx.query) {
            var seen = {};
            var recentItems = [];
            for (i = 0; i < ctx.historyKeys.length; i++) {
                var historyItem = getItemByKey(ctx.items, ctx.historyKeys[i]);
                if (historyItem) recentItems.push(historyItem);
            }

            var favoriteItems = [];
            for (i = 0; i < ctx.favoriteIds.length; i++) {
                favoriteItems.push(ctx.getItemForFavorite(ctx.favoriteIds[i]));
            }

            var currentItems = [];
            for (i = 0; i < ctx.items.length; i++) {
                if (ctx.items[i].tab === ctx.activeTab) currentItems.push(ctx.items[i]);
            }

            var browseItems = ctx.items.slice(0);
            browseItems.sort(function (a, b) {
                var tabCompare = ctx.getTabLabel(a.tab).localeCompare(ctx.getTabLabel(b.tab));
                if (tabCompare !== 0) return tabCompare;
                return (a.name || "").localeCompare(b.name || "");
            });

            addPaletteSection(sections, "Recent", recentItems, ctx, function (item) {
                return { isRecent: true, isCurrent: item.tab === ctx.activeTab };
            }, seen);

            addPaletteSection(sections, "Favorites", favoriteItems, ctx, function (item, key) {
                return { isRecent: !!historyLookup[key], isCurrent: item.tab === ctx.activeTab };
            }, seen);

            addPaletteSection(sections, ctx.activeTab ? "Current Workspace" : "Suggested Tools", currentItems, ctx, function (item, key) {
                return { isRecent: !!historyLookup[key], isCurrent: true };
            }, seen);

            addPaletteSection(sections, "Browse All", browseItems, ctx, function (item, key) {
                return { isRecent: !!historyLookup[key], isCurrent: item.tab === ctx.activeTab };
            }, seen);
            return sections;
        }

        var matches = [];
        for (i = 0; i < ctx.items.length; i++) {
            var item = ctx.items[i];
            var score = scoreCommandPaletteItem(item, ctx);
            if (!score) continue;
            matches.push(createPaletteEntry(item, ctx, {
                score: score,
                isRecent: !!historyLookup[getCommandPaletteItemKey(item)],
                isCurrent: item.tab === ctx.activeTab
            }));
        }

        matches.sort(function (a, b) {
            if (b.score !== a.score) return b.score - a.score;
            if (a.isFavorite !== b.isFavorite) return a.isFavorite ? -1 : 1;
            if (a.isCurrent !== b.isCurrent) return a.isCurrent ? -1 : 1;
            return (a.item.name || "").localeCompare(b.item.name || "");
        });

        if (matches.length) sections.push({ label: "Matching Tools", entries: matches });
        return sections;
    }

    return {
        escapeHtml: escapeHtml,
        escapeJsxDoubleQuotedString: escapeJsxDoubleQuotedString,
        createLazyDomProxy: createLazyDomProxy,
        normalizePaletteText: normalizePaletteText,
        formatPaletteLabel: formatPaletteLabel,
        getCommandPaletteItemKey: getCommandPaletteItemKey,
        scoreCommandPaletteItem: function (item, options) {
            return scoreCommandPaletteItem(item, makePaletteContext(options));
        },
        buildCommandPaletteSections: buildCommandPaletteSections
    };
});
