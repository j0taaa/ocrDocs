import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import fsp from 'fs/promises';
import os from 'os';
import { spawn } from 'child_process';
import OpenAI from 'openai';
import 'dotenv/config';

// Environment / defaults follow analyze-crefaz.js
const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.MODEL || 'google/gemma-3-4b';
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'http://10.7.7.40:1234/v1';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'lm-studio';
const PDF_DPI = Number(process.env.PDF_DPI || 72);
const MAX_PAGES = Number(process.env.MAX_PAGES || 0); // 0 = all
const DEBUG = !!process.env.DEBUG;

const UPLOADS_DIR = path.resolve(process.cwd(), '.tmp_uploads');
const TMP_PAGES_DIR = path.resolve(process.cwd(), '.tmp_pages');

function toPosix(p) {
  return p.split(path.sep).join('/');
}

async function ensureDir(directoryPath) {
  await fsp.mkdir(directoryPath, { recursive: true });
}

async function commandExists(commandName) {
  return new Promise((resolve) => {
    const test = spawn(process.env.SHELL || '/bin/sh', ['-lc', `command -v ${commandName} >/dev/null 2>&1`], { stdio: 'ignore' });
    test.on('close', (code) => resolve(code === 0));
    test.on('error', () => resolve(false));
  });
}

async function detectPdfConverter() {
  if (await commandExists('pdftoppm')) {
    return { kind: 'pdftoppm' };
  }
  if (await commandExists('magick')) {
    return { kind: 'imagemagick' };
  }
  if (await commandExists('convert')) {
    return { kind: 'imagemagick_legacy' };
  }
  return { kind: 'none' };
}

function extractPageNumber(filename) {
  const m = filename.match(/-(\d+)\.png$/i);
  return m ? Number(m[1]) : null;
}

async function listFilesSortedByPage(dir, basePrefix) {
  const names = await fsp.readdir(dir);
  const matches = names
    .filter((n) => n.startsWith(basePrefix) && n.toLowerCase().endsWith('.png'))
    .map((n) => ({ name: n, page: extractPageNumber(n) }))
    .filter((x) => x.page !== null)
    .sort((a, b) => a.page - b.page)
    .map((x) => path.join(dir, x.name));
  return matches;
}

function execSpawn(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: 'inherit', ...opts });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} exited with code ${code}`));
    });
  });
}

async function convertPdfToImages(pdfPath, outputDir) {
  await ensureDir(outputDir);
  const converter = await detectPdfConverter();
  const base = path.basename(pdfPath, path.extname(pdfPath));
  const prefix = path.join(outputDir, base);

  if (converter.kind === 'pdftoppm') {
    await execSpawn('pdftoppm', ['-png', '-rx', String(PDF_DPI), '-ry', String(PDF_DPI), pdfPath, prefix]);
    return await listFilesSortedByPage(outputDir, base);
  }
  if (converter.kind === 'imagemagick') {
    const outPattern = path.join(outputDir, `${base}-%02d.png`);
    await execSpawn('magick', ['-density', String(PDF_DPI), pdfPath, '-alpha', 'remove', '-background', 'white', outPattern]);
    return await listFilesSortedByPage(outputDir, base);
  }
  if (converter.kind === 'imagemagick_legacy') {
    const outPattern = path.join(outputDir, `${base}-%02d.png`);
    await execSpawn('convert', ['-density', String(PDF_DPI), pdfPath, '-alpha', 'remove', '-background', 'white', outPattern]);
    return await listFilesSortedByPage(outputDir, base);
  }
  throw new Error('Nenhum conversor de PDF para imagem encontrado. Instale Poppler (pdftoppm) ou ImageMagick. Ex.: brew install poppler ou brew install imagemagick');
}

async function safeRemoveDir(dir) {
  try {
    const entries = await fsp.readdir(dir);
    await Promise.all(entries.map((name) => fsp.rm(path.join(dir, name), { force: true }))).catch(() => {});
    await fsp.rmdir(dir).catch(() => {});
  } catch {}
}

async function safeUnlink(filePath) {
  try {
    await fsp.unlink(filePath);
  } catch {}
}

function buildMessagesForPdf(promptText, imagePaths) {
  const userParts = [];
  const guidance = promptText && String(promptText).trim() ? String(promptText).trim() : 'Analise o conteúdo do PDF fornecido.';
  userParts.push({ type: 'text', text: guidance });
  for (const img of imagePaths) {
    const isJpg = img.toLowerCase().endsWith('.jpg') || img.toLowerCase().endsWith('.jpeg');
    const mime = isJpg ? 'image/jpeg' : 'image/png';
    const base64 = fs.readFileSync(img, { encoding: 'base64' });
    const url = `data:${mime};base64,${base64}`;
    userParts.push({ type: 'image_url', image_url: { url } });
  }
  const messages = [
    { role: 'system', content: 'Você é um assistente útil que analisa PDFs renderizados como imagens.' },
    { role: 'user', content: userParts },
  ];
  return messages;
}

function tryParseJsonFromRaw(raw) {
  try {
    const parsed = JSON.parse(raw);
    return { ok: true, value: parsed };
  } catch (e) {
    try {
      const startIdx = raw.indexOf('{');
      const endIdx = raw.lastIndexOf('}');
      if (startIdx >= 0 && endIdx > startIdx) {
        const snippet = raw.slice(startIdx, endIdx + 1);
        const parsed = JSON.parse(snippet);
        return { ok: true, value: parsed, note: 'parsed from snippet' };
      }
    } catch {}
    return { ok: false };
  }
}

async function createChatWithOptionalSchema(openaiClient, messages, jsonSchema) {
  const base = { model: MODEL, messages, temperature: 0 };
  const tryWithSchema = async () => {
    return await openaiClient.chat.completions.create({
      ...base,
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'custom_schema',
          schema: jsonSchema,
        },
      },
    });
  };
  try {
    if (jsonSchema) {
      return await tryWithSchema();
    }
    return await openaiClient.chat.completions.create(base);
  } catch (err) {
    const msg = err?.response?.data || err?.message || '';
    const msgText = typeof msg === 'string' ? msg : JSON.stringify(msg);
    if (jsonSchema && (msgText.includes('response_format') || msgText.includes('unsupported') || msgText.includes('must be'))) {
      return await openaiClient.chat.completions.create(base);
    }
    return await openaiClient.chat.completions.create(base);
  }
}

// Express app setup
await ensureDir(UPLOADS_DIR);
await ensureDir(TMP_PAGES_DIR);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => cb(null, `upload-${Date.now()}${path.extname(file.originalname).toLowerCase()}`),
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const isPdf = file.mimetype === 'application/pdf' || path.extname(file.originalname).toLowerCase() === '.pdf';
    if (!isPdf) return cb(new Error('Somente arquivos PDF são permitidos.'));
    cb(null, true);
  },
});

const app = express();
app.use(express.json({ limit: '10mb' }));

app.get('/health', (req, res) => {
  res.json({ status: 'ok', model: MODEL });
});

// Simple proxy to the underlying OpenAI-compatible API
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const openaiClient = new OpenAI({ baseURL: OPENAI_BASE_URL, apiKey: OPENAI_API_KEY });
    const incoming = req.body || {};
    const payload = { ...incoming, model: incoming?.model || MODEL, stream: false };
    const completion = await openaiClient.chat.completions.create(payload);
    return res.status(200).json(completion);
  } catch (err) {
    const message = err?.response?.data || err?.message || String(err);
    try {
      if (typeof message === 'string') return res.status(500).json({ error: message });
      return res.status(500).json(message);
    } catch {}
    return res.status(500).json({ error: 'Proxy request failed.' });
  }
});

app.post('/v1/analyze', upload.single('file'), async (req, res) => {
  const start = Date.now();
  let uploadedPath = req.file?.path;
  const maxPagesParam = Number(req.body?.maxPages || req.query?.maxPages || 0);
  const limitPages = maxPagesParam > 0 ? maxPagesParam : (MAX_PAGES > 0 ? MAX_PAGES : 0);

  // Prompts
  const pagePrompt = (req.body?.pagePrompt || req.body?.prompt || '').toString();
  const finalPromptRaw = (req.body?.finalPrompt || '').toString();
  const finalPrompt = finalPromptRaw && finalPromptRaw.trim()
    ? finalPromptRaw
    : 'Combine os resultados extraídos de cada página em um único JSON final, mantendo consistência e removendo duplicidades. Responda SOMENTE em JSON válido.';

  if (!uploadedPath) {
    return res.status(400).json({ error: 'Arquivo PDF é obrigatório (campo form-data: file).' });
  }
  if (!pagePrompt || !String(pagePrompt).trim()) {
    await safeUnlink(uploadedPath);
    return res.status(400).json({ error: 'Parâmetro "pagePrompt" é obrigatório (ou use "prompt" para compatibilidade).' });
  }

  // Optional JSON schemas
  let pageSchema = null;
  if (typeof req.body?.pageSchema === 'string' && req.body.pageSchema.trim()) {
    try {
      pageSchema = JSON.parse(req.body.pageSchema);
    } catch (e) {
      await safeUnlink(uploadedPath);
      return res.status(400).json({ error: 'pageSchema fornecido não é um JSON válido.' });
    }
  } else if (typeof req.body?.schema === 'string' && req.body.schema.trim()) {
    // backward compat: treat previous "schema" as per-page schema
    try {
      pageSchema = JSON.parse(req.body.schema);
    } catch (e) {
      await safeUnlink(uploadedPath);
      return res.status(400).json({ error: 'schema (legado) fornecido não é um JSON válido.' });
    }
  }

  let finalSchema = null;
  if (typeof req.body?.finalSchema === 'string' && req.body.finalSchema.trim()) {
    try {
      finalSchema = JSON.parse(req.body.finalSchema);
    } catch (e) {
      await safeUnlink(uploadedPath);
      return res.status(400).json({ error: 'finalSchema fornecido não é um JSON válido.' });
    }
  }

  const openaiClient = new OpenAI({ baseURL: OPENAI_BASE_URL, apiKey: OPENAI_API_KEY });

  const baseName = path.basename(uploadedPath, path.extname(uploadedPath));
  const docTmpDir = path.join(TMP_PAGES_DIR, `${baseName}-${Date.now()}`);
  try {
    // Convert PDF -> images
    const images = await convertPdfToImages(uploadedPath, docTmpDir);
    const limited = limitPages > 0 ? images.slice(0, limitPages) : images;
    if (limited.length === 0) {
      throw new Error('Falha ao converter PDF em imagens.');
    }

    if (DEBUG) {
      console.error(`[DEBUG] Processando ${limited.length} página(s) com pagePrompt.`);
    }

    // Phase 1: per-page inference
    const perPageResults = [];
    for (let i = 0; i < limited.length; i++) {
      const pagePath = limited[i];
      const messages = buildMessagesForPdf(pagePrompt, [pagePath]);
      if (pageSchema) {
        messages.unshift({ role: 'system', content: 'Responda SOMENTE em JSON válido estritamente conforme o schema fornecido.' });
      }
      const response = await createChatWithOptionalSchema(openaiClient, messages, pageSchema);
      const raw = response?.choices?.[0]?.message?.content || '';
      if (pageSchema) {
        const parsedAttempt = tryParseJsonFromRaw(raw);
        if (parsedAttempt.ok) {
          perPageResults.push(parsedAttempt.value);
        } else {
          // if schema expected but parsing failed, push raw as fallback
          perPageResults.push({ _unparsed: raw });
        }
      } else {
        perPageResults.push(raw);
      }
    }

    // Phase 2: final aggregation
    if (DEBUG) {
      console.error('[DEBUG] Agregando resultados das páginas com finalPrompt.');
    }

    const userParts = [];
    userParts.push({ type: 'text', text: finalPrompt });
    userParts.push({ type: 'text', text: 'Resultados por página (JSON):' });
    userParts.push({ type: 'text', text: JSON.stringify(perPageResults) });

    const finalMessages = [
      { role: 'system', content: 'Você é um assistente útil que combina resultados de múltiplas páginas.' },
      { role: 'user', content: userParts },
    ];
    if (finalSchema) {
      finalMessages.unshift({ role: 'system', content: 'Responda SOMENTE em JSON válido estritamente conforme o schema fornecido.' });
    }

    const finalResponse = await createChatWithOptionalSchema(openaiClient, finalMessages, finalSchema);
    const finalRaw = finalResponse?.choices?.[0]?.message?.content || '';

    if (finalSchema) {
      const parsedAttempt = tryParseJsonFromRaw(finalRaw);
      if (parsedAttempt.ok) {
        return res.json({ model: MODEL, pagesProcessed: limited.length, result: parsedAttempt.value, ms: Date.now() - start, note: parsedAttempt.note });
      }
      return res.status(502).json({ error: 'Modelo retornou saída não-JSON na agregação final.', raw: finalRaw });
    }

    // Without finalSchema, return the free-text final answer
    return res.json({ model: MODEL, pagesProcessed: limited.length, content: finalRaw, ms: Date.now() - start });
  } catch (err) {
    const message = err?.message || String(err);
    return res.status(500).json({ error: message });
  } finally {
    await safeUnlink(uploadedPath).catch(() => {});
    await safeRemoveDir(docTmpDir).catch(() => {});
  }
});

app.listen(PORT, () => {
  console.log(`API listening on http://localhost:${PORT}`);
  console.log('Modelo (LM Studio OpenAI):', MODEL);
  if (OPENAI_BASE_URL) console.log('OPENAI_BASE_URL:', OPENAI_BASE_URL);
});


