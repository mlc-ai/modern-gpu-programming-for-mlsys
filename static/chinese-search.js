/* Split Chinese search queries using terms already present in Sphinx's index. */
var splitQuery = (query) => {
  const chunks = query.match(
    /[\p{Script=Han}]+|[\p{Letter}\p{Number}_\p{Emoji_Presentation}]+/gu,
  ) || [];

  const hasIndex = typeof Search !== "undefined" && Search._index;
  if (!hasIndex) return chunks;

  const isKnownTerm = (term) =>
    Object.prototype.hasOwnProperty.call(Search._index.terms, term) ||
    Object.prototype.hasOwnProperty.call(Search._index.titleterms, term);

  return chunks.flatMap((chunk) => {
    if (!/^\p{Script=Han}+$/u.test(chunk)) return [chunk];

    const chars = Array.from(chunk);
    const terms = [];
    let start = 0;

    while (start < chars.length) {
      let match = null;
      for (let end = chars.length; end > start + 1; end -= 1) {
        const candidate = chars.slice(start, end).join("");
        if (isKnownTerm(candidate)) {
          match = candidate;
          break;
        }
      }

      if (match) {
        terms.push(match);
        start += Array.from(match).length;
      } else {
        start += 1;
      }
    }

    return terms.length ? terms : [chunk];
  });
};
