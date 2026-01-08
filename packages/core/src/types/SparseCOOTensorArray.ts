class SparseCOOTensorArray {
  public static readonly BYTES_PER_ELEMENT = 1;
  public readonly BYTES_PER_ELEMENT = 1;
  public readonly length: number;
  private data: Map<number, number>;

  constructor(length: number, entries?: Iterable<[number, number]>) {
    this.length = length;
    this.data = new Map(entries);

    return new Proxy(this, {
      get: (target, prop) => {
        if (prop === 'length') {
          return target.length;
        }
        if (prop === 'byteLength') {
          return target.byteLength;
        }
        if (prop === 'BYTES_PER_ELEMENT') {
          return target.BYTES_PER_ELEMENT;
        }
        if (prop === 'set') {
          return target.set.bind(target);
        }
        if (prop === 'subarray') {
          return target.subarray.bind(target);
        }
        if (prop === 'values') {
          return target.values.bind(target);
        }
        if (prop === 'entries') {
          return target.entries.bind(target);
        }
        if (prop === 'keys') {
          return target.keys.bind(target);
        }
        if (prop === Symbol.iterator) {
          return target[Symbol.iterator].bind(target);
        }
        if (prop === 'constructor') {
          return SparseCOOTensorArray;
        }

        if (typeof prop === 'string') {
          const index = Number(prop);
          if (Number.isInteger(index) && index >= 0) {
            return target.get(index);
          }
        }

        // @ts-ignore
        return target[prop];
      },
      set: (target, prop, value) => {
        if (typeof prop === 'string') {
          const index = Number(prop);
          if (Number.isInteger(index) && index >= 0) {
            target.set(index, value as number);
            return true;
          }
        }
        // @ts-ignore
        target[prop] = value;
        return true;
      },
    });
  }

  get byteLength(): number {
    // Approximate memory by counting stored entries (1 byte each for label values).
    return this.data.size * this.BYTES_PER_ELEMENT;
  }

  private get(index: number): number {
    if (index >= this.length || index < 0) {
      return 0;
    }
    return this.data.get(index) ?? 0;
  }

  private setValue(index: number, value: number): void {
    if (index >= this.length || index < 0) {
      return;
    }

    if (!value) {
      this.data.delete(index);
    } else {
      this.data.set(index, value);
    }
  }

  public set(values: ArrayLike<number> | number, offset = 0): void {
    if (typeof values === 'number') {
      this.setValue(offset, values);
      return;
    }

    for (let i = 0; i < values.length && offset + i < this.length; i++) {
      this.setValue(offset + i, values[i]);
    }
  }

  public subarray(begin = 0, end = this.length): SparseCOOTensorArray {
    const start = Math.max(begin, 0);
    const stop = Math.min(end, this.length);
    const resultLength = Math.max(stop - start, 0);
    const entries: Array<[number, number]> = [];

    for (const [index, value] of this.data.entries()) {
      if (index >= start && index < stop) {
        entries.push([index - start, value]);
      }
    }

    return new SparseCOOTensorArray(resultLength, entries);
  }

  public *keys(): IterableIterator<number> {
    for (let i = 0; i < this.length; i++) {
      yield i;
    }
  }

  public *entries(): IterableIterator<[number, number]> {
    for (let i = 0; i < this.length; i++) {
      yield [i, this.get(i)];
    }
  }

  public values(): IterableIterator<number> {
    return this[Symbol.iterator]();
  }

  public [Symbol.iterator](): IterableIterator<number> {
    let current = 0;
    const length = this.length;
    const getter = this.get.bind(this);

    return {
      next(): IteratorResult<number> {
        if (current < length) {
          const value = getter(current);
          current++;
          return { value, done: false };
        }

        return { value: undefined, done: true } as IteratorResult<number>;
      },
      [Symbol.iterator](): IterableIterator<number> {
        return this;
      },
    } as IterableIterator<number>;
  }
}

export default SparseCOOTensorArray;
export { SparseCOOTensorArray };
