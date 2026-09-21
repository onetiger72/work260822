// Narrow CSV replacement. Uses the bundled Artifact Tool for cell authoring.
// Usage: bundled-node replace_prediction_csv.mjs --inspect | --check | --apply
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';

const base = path.dirname(fileURLToPath(import.meta.url));
const cache = path.join(base, 'analysis_cache');
const requireBundled = createRequire(path.join(cache, 'dependency_resolution.cjs'));
const { Workbook } = await import(pathToFileURL(requireBundled.resolve('@oai/artifact-tool')).href);
const sourcePath = path.join(base, '당첨예상번호.csv');
const inputPath = path.join(base, 'regenerated_1241.json');
const targetDraw = 1241;
const mode = process.argv[2] || '--inspect';
if (!['--inspect', '--check', '--apply'].includes(mode)) throw new Error('Use --inspect, --check, or --apply.');

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const requireTrue = (condition, message) => { if (!condition) throw new Error(message); };
const cellText = (value) => value === null || value === undefined ? '' : String(value);
const csvField = (value) => {
  const text = cellText(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const originalBytes = await fs.readFile(sourcePath);
requireTrue(originalBytes.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf])), 'Source must retain UTF-8 BOM.');
const text = originalBytes.toString('utf8').replace(/^\uFEFF/, '');
// This file has no quoted multi-line values. Refuse unknown raw layouts rather
// than risk changing unrelated rows when reusing the raw physical lines.
const rawLines = text.match(/[^\r\n]*(?:\r\n|\n|\r|$)/g).filter((line) => line.length > 0);
const wb = await Workbook.fromCSV(text, { sheetName: 'Predictions' });
const sheet = wb.worksheets.getItem('Predictions');
const original = sheet.getUsedRange().values;
requireTrue(original.length === 41 && rawLines.length === 41, 'Expected a header and 40 data rows.');
const header = original[0].map(cellText);
requireTrue(header.length === 18 && header[0] === '예측회차' && header[8] === '종합점수', 'Unexpected CSV header.');
requireTrue(original.every((row) => row.length === 18), 'Unexpected CSV row width.');
const targetIndexes = original.flatMap((row, index) => index > 0 && Number(row[0]) === targetDraw ? [index] : []);
requireTrue(targetIndexes.length === 10, 'Expected exactly 10 existing 1241 rows.');
requireTrue(targetIndexes.every((index, offset) => index === 31 + offset), 'Expected existing target rows at the end.');
const inspect = await wb.inspect({ kind: 'table', range: 'Predictions!A31:R41', include: 'values,formulas', tableMaxRows: 11, tableMaxCols: 18, maxChars: 6000 });

await fs.mkdir(cache, { recursive: true });
async function renderTo(name) {
  const preview = await wb.render({ sheetName: 'Predictions', range: 'A31:R41', scale: 1.5, format: 'png' });
  await fs.writeFile(path.join(cache, name), new Uint8Array(await preview.arrayBuffer()));
}

if (mode === '--inspect') {
  console.log(inspect.ndjson);
  await renderTo('prediction_csv_before.png');
  console.log(JSON.stringify({ mode, original_sha256: hash(originalBytes), rows: original.length - 1, target_rows: targetIndexes.length }));
  process.exit(0);
}

const payload = JSON.parse(await fs.readFile(inputPath, 'utf8'));
requireTrue(!Array.isArray(payload) && payload.target_draw === targetDraw, 'Input must identify target_draw 1241.');
requireTrue(typeof payload.original_csv_sha256 === 'string' && payload.original_csv_sha256 === hash(originalBytes), 'Source CSV differs from the regenerated prediction input; stop and regenerate.');
requireTrue(JSON.stringify(payload.columns) === JSON.stringify(header), 'Input columns must exactly match the existing CSV header.');
const records = payload.rows;
requireTrue(Array.isArray(records) && records.length === 10, 'Input must contain exactly 10 rows.');
const seen = new Set();
const newRows = records.map((record, offset) => {
  requireTrue(record && typeof record === 'object' && header.every((column) => Object.hasOwn(record, column)), 'Every input row must contain all 18 CSV fields.');
  requireTrue(record['예측회차'] === targetDraw, 'Every input row must target 1241.');
  requireTrue(header.slice(9).every((column) => record[column] === '' || record[column] === null), 'Input future actual values must be blank.');
  const setNumber = record['세트'];
  requireTrue(Number.isSafeInteger(setNumber) && setNumber === offset + 1, 'Set IDs must be ordered 1 through 10.');
  const numbers = Array.from({ length: 6 }, (_, i) => record[`번호${i + 1}`]);
  requireTrue(numbers.every((n) => Number.isSafeInteger(n) && n >= 1 && n <= 45), 'Each number must be an integer from 1 through 45.');
  requireTrue(numbers.every((n, i) => i === 0 || numbers[i - 1] < n), 'Ticket numbers must be unique and ascending.');
  requireTrue(!seen.has(numbers.join(',')), 'Duplicate target ticket.');
  seen.add(numbers.join(','));
  const score = record['종합점수'];
  requireTrue(typeof score === 'number' && Number.isFinite(score), 'Missing or nonnumeric score.');
  return [targetDraw, setNumber, ...numbers, score, ...Array(9).fill(null)];
});

if (mode === '--check') {
  console.log(JSON.stringify({ mode, target_draw: targetDraw, source_sha256: hash(originalBytes), input_sha256: hash(await fs.readFile(inputPath)), data_rows: original.length - 1, target_rows: newRows.length, preserved_rows: 30, all_input_actual_fields_blank: true, unique_ascending_tickets: true }));
  process.exit(0);
}

// The only spreadsheet authoring operation touches the existing target cells.
sheet.getRange('A32:R41').values = newRows;
wb.recalculate();
const updated = sheet.getUsedRange().values;
requireTrue(updated.length === 41, 'Row count changed.');
for (let i = 0; i <= 30; i++) {
  requireTrue(JSON.stringify(updated[i]) === JSON.stringify(original[i]), `Unrelated row ${i + 1} changed.`);
}
for (let i = 31; i <= 40; i++) {
  requireTrue(JSON.stringify(updated[i].map(cellText)) === JSON.stringify(newRows[i - 31].map(cellText)), `Target row ${i + 1} mismatch.`);
  requireTrue(updated[i].slice(9).every((v) => cellText(v) === ''), 'Future actual values must be blank.');
}
console.log((await wb.inspect({ kind: 'table', range: 'Predictions!A32:R41', include: 'values,formulas', tableMaxRows: 10, tableMaxCols: 18, maxChars: 6500 })).ndjson);
console.log((await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 20 }, summary: 'Final CSV cell error scan' })).ndjson);
await renderTo('prediction_csv_after.png');

// Artifact Tool has no documented CSV exporter in this installed reference.
// Serialize its authored cell values; keep unrelated physical lines verbatim.
const newline = rawLines[0].endsWith('\r\n') ? '\r\n' : rawLines[0].endsWith('\r') ? '\r' : '\n';
const changedLines = updated.slice(31).map((row) => row.map(csvField).join(',') + newline);
const replacement = Buffer.from('\uFEFF' + rawLines.slice(0, 31).join('') + changedLines.join(''), 'utf8');
const reopened = await Workbook.fromCSV(replacement.toString('utf8').replace(/^\uFEFF/, ''), { sheetName: 'Verification' });
const verified = reopened.worksheets.getItem('Verification').getUsedRange().values;
requireTrue(verified.length === 41 && verified.slice(1).filter((row) => Number(row[0]) === targetDraw).length === 10, 'Final CSV row count mismatch.');
requireTrue(JSON.stringify(verified.slice(0, 31)) === JSON.stringify(original.slice(0, 31)), 'Preserved rows differ after CSV serialization.');
requireTrue(JSON.stringify(verified.slice(31).map((row) => row.map(cellText))) === JSON.stringify(newRows.map((row) => row.map(cellText))), 'Reopened target rows differ.');
const preservedPrefix = Buffer.from('\uFEFF' + rawLines.slice(0, 31).join(''), 'utf8');
requireTrue(replacement.subarray(0, preservedPrefix.length).equals(originalBytes.subarray(0, preservedPrefix.length)), 'Earlier raw CSV bytes changed.');
requireTrue(hash(await fs.readFile(sourcePath)) === hash(originalBytes), 'CSV changed concurrently; stop and re-inspect.');

const stamp = new Date().toISOString().replaceAll(':', '').replaceAll('.', '-');
const backupDir = path.join(base, 'backups');
await fs.mkdir(backupDir, { recursive: true });
const backupPath = path.join(backupDir, `당첨예상번호_before_1241_${stamp}.csv`);
await fs.writeFile(backupPath, originalBytes, { flag: 'wx' });
const temporaryPath = path.join(base, `.prediction_csv_${stamp}.tmp`);
await fs.writeFile(temporaryPath, replacement, { flag: 'wx' });
await fs.rename(temporaryPath, sourcePath);
requireTrue(hash(await fs.readFile(sourcePath)) === hash(replacement), 'Saved file hash mismatch.');
const result = {
  target_draw: targetDraw, data_rows: 40, replaced_rows: 10,
  preserved_rows: 30, earlier_rows_byte_identical: true, actual_fields_blank: true,
  utf8_bom_preserved: true, source_sha256: hash(originalBytes), output_sha256: hash(replacement),
  backup_path: backupPath, output_path: sourcePath, authoring: '@oai/artifact-tool',
  csv_export: 'standard CSV encoding of authored values; original unrelated physical lines retained',
};
await fs.writeFile(path.join(base, 'csv_replacement_verification.json'), JSON.stringify(result, null, 2), 'utf8');
console.log(JSON.stringify(result, null, 2));
