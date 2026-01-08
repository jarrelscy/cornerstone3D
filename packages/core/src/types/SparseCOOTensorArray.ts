import { tensor } from '@hoff97/tensor-js';

type TensorJsCPUTensor = InstanceType<typeof tensor.cpu.CPUTensor>;
type TensorJsSparseTensor = InstanceType<typeof tensor.sparse.SparseTensor>;

class SparseCOOTensorArray {
  public static readonly BYTES_PER_ELEMENT = 1;
  public readonly BYTES_PER_ELEMENT = 1;
  public readonly length: number;
  private indexMap = new Map<number, number>();
  private valuesTensor: TensorJsCPUTensor;
  private indicesTensor: TensorJsCPUTensor;
  private sparseTensor: TensorJsSparseTensor;

  constructor(length: number, entries?: Iterable<[number, number]>) {
    this.length = length;

    const { values, indices, indexMap } = this.buildArraysFromEntries(entries);
    this.setTensors(values, indices, indexMap);

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
        if (prop === 'getSparseTensor') {
          return target.getSparseTensor.bind(target);
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
    return (
      (this.valuesTensor.values as Uint8Array).byteLength +
      (this.indicesTensor.values as Uint32Array).byteLength
    );
  }

  public getSparseTensor(): TensorJsSparseTensor {
    return this.sparseTensor;
  }

  private buildArraysFromEntries(entries?: Iterable<[number, number]>): {
    values: Uint8Array;
    indices: Uint32Array;
    indexMap: Map<number, number>;
  } {
    const values: number[] = [];
    const indices: number[] = [];
    const indexMap = new Map<number, number>();

    if (entries) {
      for (const [index, value] of entries) {
        if (
          index < 0 ||
          index >= this.length ||
          !Number.isFinite(value) ||
          !value
        ) {
          continue;
        }

        const position = values.length;
        values.push(value);
        indices.push(index);
        indexMap.set(index, position);
      }
    }

    return {
      values: new Uint8Array(values),
      indices: new Uint32Array(indices),
      indexMap,
    };
  }

  private setTensors(
    values: Uint8Array,
    indices: Uint32Array,
    indexMap: Map<number, number>
  ): void {
    this.indexMap = indexMap;
    this.valuesTensor = new tensor.cpu.CPUTensor(
      [values.length],
      values,
      'uint8'
    );
    this.indicesTensor = new tensor.cpu.CPUTensor(
      [indices.length, 1],
      indices,
      'uint32'
    );
    this.sparseTensor = new tensor.sparse.SparseTensor(
      this.valuesTensor,
      this.indicesTensor,
      [this.length]
    );
  }

  private get(index: number): number {
    if (index >= this.length || index < 0) {
      return 0;
    }

    const position = this.indexMap.get(index);
    if (position === undefined) {
      return 0;
    }

    return (this.valuesTensor.values as Uint8Array)[position] ?? 0;
  }

  private removeAtPosition(position: number): void {
    const values = this.valuesTensor.values as Uint8Array;
    const indices = this.indicesTensor.values as Uint32Array;
    const newLength = values.length - 1;
    const nextValues = new Uint8Array(Math.max(newLength, 0));
    const nextIndices = new Uint32Array(Math.max(newLength, 0));
    const nextMap = new Map<number, number>();

    let writeIndex = 0;
    for (let i = 0; i < values.length; i++) {
      if (i === position) {
        continue;
      }
      const index = indices[i];
      nextValues[writeIndex] = values[i];
      nextIndices[writeIndex] = index;
      nextMap.set(index, writeIndex);
      writeIndex++;
    }

    this.setTensors(nextValues, nextIndices, nextMap);
  }

  private appendEntry(index: number, value: number): void {
    const values = this.valuesTensor.values as Uint8Array;
    const indices = this.indicesTensor.values as Uint32Array;
    const nextValues = new Uint8Array(values.length + 1);
    const nextIndices = new Uint32Array(indices.length + 1);

    nextValues.set(values);
    nextIndices.set(indices);
    nextValues[values.length] = value;
    nextIndices[indices.length] = index;

    const nextMap = new Map(this.indexMap);
    nextMap.set(index, values.length);
    this.setTensors(nextValues, nextIndices, nextMap);
  }

  private setValue(index: number, value: number): void {
    if (index >= this.length || index < 0) {
      return;
    }

    const position = this.indexMap.get(index);
    if (!value) {
      if (position !== undefined) {
        this.removeAtPosition(position);
      }
      return;
    }

    if (position !== undefined) {
      (this.valuesTensor.values as Uint8Array)[position] = value;
      return;
    }

    this.appendEntry(index, value);
  }

  private rebuildFromDense(values: ArrayLike<number>, offset: number): void {
    const indices: number[] = [];
    const newValues: number[] = [];
    const indexMap = new Map<number, number>();
    const maxLength = Math.min(values.length, this.length - offset);

    for (let i = 0; i < maxLength; i++) {
      const value = values[i];
      if (!value) {
        continue;
      }
      const index = offset + i;
      const position = newValues.length;
      newValues.push(value);
      indices.push(index);
      indexMap.set(index, position);
    }

    this.setTensors(
      new Uint8Array(newValues),
      new Uint32Array(indices),
      indexMap
    );
  }

  public set(values: ArrayLike<number> | number, offset = 0): void {
    if (typeof values === 'number') {
      this.setValue(offset, values);
      return;
    }

    if (offset === 0 && values.length >= this.length) {
      this.rebuildFromDense(values, 0);
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
    const values = this.valuesTensor.values as Uint8Array;
    const indices = this.indicesTensor.values as Uint32Array;

    for (let i = 0; i < indices.length; i++) {
      const index = indices[i];
      if (index >= start && index < stop) {
        entries.push([index - start, values[i]]);
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
