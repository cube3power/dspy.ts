// Mock implementation of js-pytorch for testing

// Mock implementation of js-pytorch for testing
export const nn = {
  Linear: jest.fn().mockImplementation((inputSize: number, outputSize: number) => ({
    inputSize,
    outputSize,
    forward: jest.fn().mockImplementation(x => x),
    to: jest.fn(),
    eval: jest.fn(),
    copy_: jest.fn()
  })),
  ReLU: jest.fn().mockImplementation(() => ({
    forward: jest.fn().mockImplementation(x => x),
    to: jest.fn(),
    eval: jest.fn()
  }))
};

// Flat tensor stub — must NOT recursively construct child tensors, otherwise
// `tensor(...)` overflows the stack (each child's `.add`/`.pow`/… eagerly
// builds another tensor → infinite recursion).
const tensorStub = () => ({
  shape: [1],
  dataSync: jest.fn().mockReturnValue(new Float32Array([0])),
  add: jest.fn(),
  pow: jest.fn(),
  sum: jest.fn(),
  backward: jest.fn(),
  relu: jest.fn(),
  to: jest.fn(),
  copy_: jest.fn(),
});

export const tensor = jest.fn().mockImplementation((data: number[] | Float32Array, _options?: { requiresGrad?: boolean }) => ({
  shape: Array.isArray(data) ? [data.length] : [data.byteLength / 4],
  dataSync: jest.fn().mockReturnValue(Array.isArray(data) ? new Float32Array(data) : data),
  add: jest.fn(() => tensorStub()),
  pow: jest.fn(() => tensorStub()),
  sum: jest.fn(() => tensorStub()),
  backward: jest.fn(),
  relu: jest.fn(() => tensorStub()),
  to: jest.fn(() => tensorStub()),
  copy_: jest.fn(),
}));

export const device = jest.fn().mockImplementation((type: string) => ({ type }));

export const load = jest.fn().mockImplementation(async (path: string) => ({}));
