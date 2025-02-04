
// Copyright (partly) 2021 Albert Zeyer
// This also contains some code from CPython.

// c++ -std=c++11 py-to-pickle.cpp -o py-to-pickle.bin
// c++ -std=c++11 -DLIB py-to-pickle.cpp -shared -fPIC -o libpytopickle.so
// ./py-to-pickle.bin demo.txt demo.pkl
// python3 -c "import pickle; pickle.load(open('demo.pkl', 'rb'))"

// https://github.com/python/cpython/blob/master/Modules/_pickle.c
// load_dict

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

const int protocol = 3;

/* Pickle opcodes. These must be kept updated with pickle.py.
   Extensive docs are in pickletools.py. */
enum opcode {
	MARK            = '(',
	STOP            = '.',
	POP             = '0',
	POP_MARK        = '1',
	DUP             = '2',
	FLOAT           = 'F',
	INT             = 'I',
	BININT          = 'J',
	BININT1         = 'K',
	LONG            = 'L',
	BININT2         = 'M',
	NONE            = 'N',
	PERSID          = 'P',
	BINPERSID       = 'Q',
	REDUCE          = 'R',
	STRING          = 'S',
	BINSTRING       = 'T',
	SHORT_BINSTRING = 'U',
	UNICODE         = 'V',
	BINUNICODE      = 'X',
	APPEND          = 'a',
	BUILD           = 'b',
	GLOBAL          = 'c',
	DICT            = 'd',
	EMPTY_DICT      = '}',
	APPENDS         = 'e',
	GET             = 'g',
	BINGET          = 'h',
	INST            = 'i',
	LONG_BINGET     = 'j',
	LIST            = 'l',
	EMPTY_LIST      = ']',
	OBJ             = 'o',
	PUT             = 'p',
	BINPUT          = 'q',
	LONG_BINPUT     = 'r',
	SETITEM         = 's',
	TUPLE           = 't',
	EMPTY_TUPLE     = ')',
	SETITEMS        = 'u',
	BINFLOAT        = 'G',

	/* Protocol 2. */
	PROTO       = '\x80',
	NEWOBJ      = '\x81',
	EXT1        = '\x82',
	EXT2        = '\x83',
	EXT4        = '\x84',
	TUPLE1      = '\x85',
	TUPLE2      = '\x86',
	TUPLE3      = '\x87',
	NEWTRUE     = '\x88',
	NEWFALSE    = '\x89',
	LONG1       = '\x8a',
	LONG4       = '\x8b',

	/* Protocol 3 (Python 3.x) */
	BINBYTES       = 'B',
	SHORT_BINBYTES = 'C',

	/* Protocol 4 */
	SHORT_BINUNICODE = '\x8c',
	BINUNICODE8      = '\x8d',
	BINBYTES8        = '\x8e',
	EMPTY_SET        = '\x8f',
	ADDITEMS         = '\x90',
	FROZENSET        = '\x91',
	NEWOBJ_EX        = '\x92',
	STACK_GLOBAL     = '\x93',
	MEMOIZE          = '\x94',
	FRAME            = '\x95',

	/* Protocol 5 */
	BYTEARRAY8       = '\x96',
	NEXT_BUFFER      = '\x97',
	READONLY_BUFFER  = '\x98'
};

static void _PyFloat_Pack8(double x, unsigned char *p, int le) {
	typedef enum {
		unset, ieee_big_endian_format, ieee_little_endian_format
	} float_format_type;
	static float_format_type double_format = unset;

#if __SIZEOF_DOUBLE__ == 8
	if(double_format == unset) {
		double x = 9006104071832581.0;
		if (memcmp(&x, "\x43\x3f\xff\x01\x02\x03\x04\x05", 8) == 0)
			double_format = ieee_big_endian_format;
		else if (memcmp(&x, "\x05\x04\x03\x02\x01\xff\x3f\x43", 8) == 0)
			double_format = ieee_little_endian_format;
		else {
			fprintf(stderr, "invalid double format");
			abort();
		}
	}
#else
#error invalid __SIZEOF_DOUBLE__
#endif

	const unsigned char *s = (unsigned char*)&x;
	int i, incr = 1;

	if ((double_format == ieee_little_endian_format && !le)
		|| (double_format == ieee_big_endian_format && le)) {
		p += 7;
		incr = -1;
	}

	for (i = 0; i < 8; i++) {
		*p = *s++;
		p += incr;
	}
}

static void _write_size64(char *out, size_t value) {
	size_t i;
	static_assert(sizeof(size_t) <= 8, "sizeof size_t != 8");

	for (i = 0; i < sizeof(size_t); i++) {
		out[i] = (unsigned char)((value >> (8 * i)) & 0xff);
	}
	for (i = sizeof(size_t); i < 8; i++) {
		out[i] = 0;
	}
}

/**
 * Encode a code point using UTF-8
 * Code adopted for C++
 *
 * @author Ondrej Hruska <ondra@ondrovo.com>
 * @license MIT
 *
 * @param utf - code point 0-0x10FFFF
 * @return number of bytes on success, 0 on failure (also produces U+FFFD, which uses 3 bytes)
 */
int utf8_encode(std::string& out, uint32_t utf)
{
  if (utf <= 0x7F) {
	// Plain ASCII
	out.push_back((char) utf);
	return 1;
  }
  else if (utf <= 0x07FF) {
	// 2-byte unicode
	out.push_back((char) (((utf >> 6) & 0x1F) | 0xC0));
	out.push_back((char) (((utf >> 0) & 0x3F) | 0x80));
	return 2;
  }
  else if (utf <= 0xFFFF) {
	// 3-byte unicode
	out.push_back((char) (((utf >> 12) & 0x0F) | 0xE0));
	out.push_back((char) (((utf >>  6) & 0x3F) | 0x80));
	out.push_back((char) (((utf >>  0) & 0x3F) | 0x80));
	return 3;
  }
  else if (utf <= 0x10FFFF) {
	// 4-byte unicode
	out.push_back((char) (((utf >> 18) & 0x07) | 0xF0));
	out.push_back((char) (((utf >> 12) & 0x3F) | 0x80));
	out.push_back((char) (((utf >>  6) & 0x3F) | 0x80));
	out.push_back((char) (((utf >>  0) & 0x3F) | 0x80));
	return 4;
  }
  else {
	// error - use replacement character
	out.push_back((char) 0xEF);
	out.push_back((char) 0xBF);
	out.push_back((char) 0xBD);
	return 0;
  }
}

struct Reader {
	virtual ~Reader() {}
	virtual bool valid() = 0;
	virtual size_t pos() = 0;
	virtual int read_next_char() = 0;
};

struct FileReader : Reader {
	FILE* fp;

	FileReader(const char* filename) {
		fp = fopen(filename, "r");
		if(fp) flockfile(fp);
	}
	virtual ~FileReader() { fclose(fp); };
	virtual bool valid() { return fp; }
	virtual size_t pos() { return ftell(fp); } // rarely called
	virtual int read_next_char() {
		return getc_unlocked(fp);
	}
};

struct MemReader : Reader {
	const char* data;
	size_t size;
	size_t p;

	MemReader(const char* data_, size_t size_) : data(data_), size(size_), p(0) {}
	virtual bool valid() { return true; }
	virtual size_t pos() { return p; }
	virtual int read_next_char() {
		if(p >= size)
			return EOF;
		unsigned char c = (unsigned char) data[p];
		++p;
		return c;
	}
};

struct Writer {
	virtual ~Writer() {}
	virtual bool valid() = 0;
	virtual size_t pos() = 0;
	virtual void seek(size_t pos) = 0;
	virtual void write_char(char c) = 0;
	virtual void write_data(const char* data, size_t len) = 0;
};

struct FileWriter : Writer {
	FILE* fp;
	size_t out_pos;

	FileWriter(const char* filename) : out_pos(0) {
		fp = fopen(filename, "wb");
		if(fp) flockfile(fp);
	}
	virtual ~FileWriter() { fclose(fp); }
	virtual bool valid() { return fp; }
	virtual size_t pos() { return out_pos; }
	virtual void seek(size_t pos) {
		fseek(fp, pos, SEEK_SET);
		out_pos = pos;
	}
	virtual void write_char(char c) {
		putc_unlocked(c, fp);
		++out_pos;
	}
	virtual void write_data(const char* data, size_t len) {
#ifdef __linux__
		fwrite_unlocked(data, len, 1, fp);
#else
		fwrite(data, len, 1, fp);
#endif
		out_pos += len;
	}
};

struct MemWriter : Writer {
	char* data;
	size_t size;
	size_t p;
	bool got_error;

	MemWriter(char* data_, size_t size_) : data(data_), size(size_), p(0), got_error(false) {}
	virtual bool valid() { return true; }
	virtual size_t pos() { return p; }
	virtual void seek(size_t pos) { p = pos; }
	virtual void write_char(char c) {
		if(p >= size) {
			_overflow_error(1);
			return;
		}
		data[p] = c;
		++p;
	}
	virtual void write_data(const char* data_, size_t len_) {
		if(p + len_ > size) {
			_overflow_error(len_);
			return;
		}
		memcpy(data + p, data_, len_);
		p += len_;
	}

	void _overflow_error(size_t more) {
		fprintf(stderr, "MemWriter, overflowing to the buffer (pos %li, size %li, add %li)\n", p, size, more);
		got_error = true;
	}
};

typedef std::pair<int,bool> ParseRes;  // next char + parsed one item or not

class Parser {
	Reader* reader;
	Writer* writer;

public:
	bool got_error;
	Parser(Reader* reader_, Writer* writer_) : reader(reader_), writer(writer_), got_error(false) {}

private:
	void parse_error(const char* ctx, char c) {
		fprintf(stderr, "parse error: %s: char '%c' in pos %li\n", ctx, c, reader->pos());
		got_error = true;
	}

	int read_next_char() { return reader->read_next_char(); }
	void write_char(char c) { writer->write_char(c); }
	void write_data(const char* data, size_t len) { writer->write_data(data, len); }

public:
	void full_pass() {
		start();
		ParseRes res = parse();
		if(!res.second) {
			parse_error("root", res.first);
			return;
		}
		end();
	}

	void start() {
		write_char(PROTO);
		write_char(protocol);
	}
	void end() {
		write_char(STOP);
	}

	void parse_list() {
		write_char(EMPTY_LIST);
		write_char(MARK);
		while(true) {
			ParseRes res = parse();
			int c = res.first;
			if(c == ',') continue;
			else if(c == ']') break;
			else { parse_error("list", c); return; }
		}
		write_char(APPENDS);
	}

	void parse_dict_or_set() {
		int c;
		size_t count = 0;
		enum {Dict, Set} obj_type = Dict;
		enum {Key, Value} cur = Key;
		long start_out_pos = writer->pos();
		auto make_set = [&]{
			if(count != 1) {
				parse_error("dict after parsing more than one entry", c);
				return;
			}
			cur = Value;
			obj_type = Set;
			long cur_pos = writer->pos();
			writer->seek(start_out_pos);
			write_char(EMPTY_SET);
			writer->seek(cur_pos);
		};
		write_char(EMPTY_DICT);
		write_char(MARK);
		while(true) {
			ParseRes res = parse();
			c = res.first;
			if(res.second) ++count;
			if(c == ',') {
				if(count == 1) { make_set(); if(got_error) return; }
				if(cur == Key) { parse_error("dict after parsing key", c); return; }
				if(obj_type == Dict) cur = Key;
			}
			else if(c == ':') {
				if(cur != Key) { parse_error("dict expected key before", c); return; }
				if(obj_type == Set) { parse_error("set", c); return; }
				cur = Value;
			}
			else if(c == '}') {
				if(count == 1) { make_set(); if(got_error) return; }
				break;
			}
			else { parse_error("dict|set", c); return; }
		}
		if(obj_type == Dict && count % 2 != 0) { parse_error("dict, uneven count", c); return; }
		write_char((obj_type == Dict) ? SETITEMS : ADDITEMS);
	}

	void parse_str(char quote) {
		// We expect to already have utf8 here (i.e. input file is utf8).
		// This can potentially be sped up, by writing early,
		// and then filling in the size. (We might want to ignore BINUNICODE8.)
		// https://docs.python.org/3/reference/lexical_analysis.html#grammar-token-python-grammar-stringescapeseq
		int c;
		std::string buf;
		enum {Direct, EscapeInit, EscapeHex} escape_mode = Direct;
		int escape_hex_rem = 0;
		uint32_t escape_hex = 0;
		while(true) {
			c = read_next_char();
			if(c < 0) { parse_error("str, got EOF", c); return; }
			if(escape_mode == Direct) {
				if(c == quote) break;
				if(c == '\\') escape_mode = EscapeInit;
				else buf.push_back((char) c);
			}
			else if(escape_mode == EscapeInit) {
				if(c == 'x' || c == 'u' || c == 'U') {
					escape_mode = EscapeHex;
					switch(c) {
						case 'x': escape_hex_rem = 2; break; // 8 bit
						case 'u': escape_hex_rem = 4; break; // 16 bit
						case 'U': escape_hex_rem = 8; break; // 32 bit
						default: assert(false);
					}
					escape_hex = 0;
				}
				else {
					char c_;
					if(c == 'r') c_ = '\r';
					else if(c == 't') c_ = '\t';
					else if(c == 'n') c_ = '\n';
					else if(c == '\\' || c == '"' || c == '\'' || c == '\n') c_ = char(c);
					else { parse_error("str escaped", c); return; }
					buf.push_back(c_);
					escape_mode = Direct;
				}
			}
			else if(escape_mode == EscapeHex) {
				int h;
				if(c >= '0' && c <= '9') h = c - '0';
				else if(c >= 'a' && c <= 'f') h = c - 'a' + 10;
				else { parse_error("str hex escaped", c); return; }
				escape_hex *= 16;
				escape_hex += h;
				escape_hex_rem--;
				if(escape_hex_rem <= 0) {
					if(utf8_encode(buf, escape_hex) <= 0)
					{ parse_error("utf8 encode", c); return; }
					escape_mode = Direct;
				}
			}
		}

		char header[9];
		int len;
		size_t size = buf.length();
		if(size <= 0xff) {
			header[0] = SHORT_BINUNICODE;
			header[1] = (unsigned char)(size & 0xff);
			len = 2;
		}
		else if(size <= 0xffffffffUL) {
			header[0] = BINUNICODE;
			header[1] = (unsigned char)(size & 0xff);
			header[2] = (unsigned char)((size >> 8) & 0xff);
			header[3] = (unsigned char)((size >> 16) & 0xff);
			header[4] = (unsigned char)((size >> 24) & 0xff);
			len = 5;
		}
		else {
			header[0] = BINUNICODE8;
			_write_size64(header + 1, size);
			len = 9;
		}
		write_data(header, len);
		write_data(buf.data(), size);
	}

	int parse_num(char first) {
		int c;
		std::string buf;
		bool is_float = false;
		buf.push_back(first);

		while(true) {
			c = read_next_char();
			if(c < 0) break;

			if((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.') {
				if(c == '.')
					is_float = true;
				buf.push_back(c);
			}
			else
				break;
		}

		if(is_float) {
			double val = std::stod(buf);
			char pdata[9];
			pdata[0] = BINFLOAT;
			_PyFloat_Pack8(val, (unsigned char *)&pdata[1], 0);
			write_data(pdata, sizeof(pdata));
		}
		else {
			int len;
			long val = std::stol(buf);
			char pdata[8];
			pdata[1] = (unsigned char)(val & 0xff);
			pdata[2] = (unsigned char)((val >> 8) & 0xff);
			pdata[3] = (unsigned char)((val >> 16) & 0xff);
			pdata[4] = (unsigned char)((val >> 24) & 0xff);

			if ((pdata[4] != 0) || (pdata[3] != 0)) {
				pdata[0] = BININT;
				len = 5;
			}
			else if (pdata[2] != 0) {
				pdata[0] = BININT2;
				len = 3;
			}
			else {
				pdata[0] = BININT1;
				len = 2;
			}
			write_data(pdata, len);
		}
		return c;
	}

	void parse_none(char first) {
		if (!('o' == read_next_char() && 'n' == read_next_char() && 'e' == read_next_char())) {
			parse_error("expected 'one' after N", first);
			return;
		}
		write_char(NONE);
	}

	ParseRes parse() {
		int c;
		bool had_one_item = false;

		while(true) {
			c = read_next_char();
			if(c < 0) break;

			if(isspace(c))
				continue;

			// (continued strings not yet supported...)
			if(had_one_item)
				break;

			if(c == '\'' || c == '"') {
				parse_str(c);
				had_one_item = true;
			}
			else if(c == '[') {
				parse_list();
				had_one_item = true;
			}
			else if(c == '{') {
				parse_dict_or_set();
				had_one_item = true;
			}
			else if((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.') {
				c = parse_num(c);
				had_one_item = true;
				if(c < 0 || !isspace(c))
					break;
			}
			else if(c == 'N') {
				parse_none(c);
				had_one_item = true;
			}
			else
				// some unexpected char
				break;
		}

		return ParseRes(c, had_one_item);
	}
};

#ifdef LIB
extern "C"
int py_to_pickle(const char* in, size_t in_len, char* out, size_t out_len) {
	MemReader reader(in, in_len);
	MemWriter writer(out, out_len);
	Parser parser(&reader, &writer);
	parser.full_pass();
	if(parser.got_error)
		return 1;
	if(writer.got_error)
		return 2;
	return 0;
}

#else  // LIB

int main(int argc, char** argv) {
	if(argc <= 2) {
		printf("usage: %s <in-py-file> <out-pickle-file>\n", argv[0]);
		return -1;
	}

	FileReader reader(argv[1]);
	if(!reader.valid()) {
		fprintf(stderr, "cannot open input file %s\n", argv[1]);
		return 1;
	}

	FileWriter writer(argv[2]);
	if(!writer.valid()) {
		fprintf(stderr, "cannot open output file %s\n", argv[2]);
		return 1;
	}

	Parser parser(&reader, &writer);
	parser.full_pass();
	return 0;
}
#endif
